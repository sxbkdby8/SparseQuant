import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Hardware Constants ---
# ZCU 102 On-Chip Resources
ZCU102_LUT_TOTAL = 274080
ZCU102_DSP_TOTAL = 2520
ZCU102_BRAM_TOTAL = 912  # 36Kb blocks
RESOURCE_BUDGET_RATIO = 0.8 # Allow up to 80% utilization, but frequency will drop

# Simplified view: 18K/36K blocks. Assuming 36K usage configuration:
# Depth 1024 x Width 36 (32 data + 4 parity) is a common aspect ratio.
BRAM_PHYSICAL_DEPTH = 1024
BRAM_PHYSICAL_WIDTH_BITS = 32 

# Energy Estimation 
E_DRAM_PJ_BIT = 15.0   # LPDDR4 access
E_REG_PJ_BIT = 0.05    # Register file
E_MAC_PJ = 0.2         # INT 8 MAC 

E_SRAM_BASE_PJ_BIT = 0.1  

P_STATIC_BASE_W = 3.5
P_STATIC_UTIL_FACTOR = 4.0

FREQ_BASE_MHZ = 350.0   # Max theoretical Fmax at low utilization
FREQ_MIN_MHZ = 100.0    # Floor Fmax at high congestion

# Search Space
ARRAY_SIZE_SRART = 4
ARRAY_SIZE_END = 48
ARRAY_SIZE_STEP = 4

class SparsityMode(Enum):
    SPARSE_1_4 = 1
    SPARSE_2_4 = 2
    SPARSE_3_4 = 3
    SPARSE_4_4 = 4

@dataclass
class MatrixConfig:
    M: int
    N: int
    K: int
    Head: int = 1
    
@dataclass
class HardwareConfig:
    array_rows: int       
    array_cols: int
    
    data_width_bits: int = 8  # INT8
    accum_width_bits: int = 32
    mask_bits: int = 3  # mask bits for each compressed weight mask
    
    use_double_buffering: bool = True 
    
    # Buffer depths (Logical depth per bank)
    dense_buffer_depth: int = 512  
    sparse_buffer_depth: int = 512
    output_buffer_depth: int = 512
    
    dram_bw_gbps: float = 12.0 * 0.8
    
    def __post_init__(self):
        self.dense_buffer_bank = self.array_rows
        self.sparse_buffer_bank = self.array_cols
        self.output_buffer_bank = self.array_cols
    
    @property   
    def pe_count(self):
        return self.array_rows * self.array_cols

    @property
    def dense_bank_width_bytes(self):
        return 4 * (self.data_width_bits // 8) 

    @property
    def input_buffer_size_bytes(self):
        # 2x for Double Buffering (Ping-Pong)
        factor = 2 if self.use_double_buffering else 1
        return self.dense_buffer_depth * self.dense_buffer_bank * self.dense_bank_width_bytes * factor
    
    @property
    def weight_buffer_size_bytes(self):
        factor = 2 if self.use_double_buffering else 1
        # 2x width roughly to account for Sparse Metadata overhead estimation in size
        return self.sparse_buffer_depth * self.sparse_buffer_bank * 2 * factor
    
    @property
    def output_buffer_size_bytes(self):
        return self.output_buffer_depth * self.output_buffer_bank * (self.data_width_bits // 8)


@dataclass
class WorkloadConfig:
    layer_name: str
    input_m:int
    input_n:int
    input_k:int
    input_head:int
    sparse_mode: SparsityMode
    
@dataclass 
class PerfMetrics:
    target_workload: WorkloadConfig
    hw_config: HardwareConfig
    
    cycles_total: int
    energy_uj: float
    throughput_gops: float = 0.0
    frequency_used_mhz: float = 0.0 # To track what freq was decided
    dram_access_mb: float = 0.0
    
    bound_by: str = "Compute"
    utilization_resource: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""
    
    
class FlexSparseDSE:
    def __init__(self):
        pass
    
    def calculate_bram_usage(self, hw_cfg: HardwareConfig) -> int:
        """
        Calculates physical BRAM blocks needed based on Width and Depth requirements,
        rather than just total bits.
        """
        def get_blocks_for_buffer(logical_depth, logical_width_bits, num_banks, is_pipo):
            # Effective depth required (x2 if ping-pong)
            req_depth = logical_depth * (2 if is_pipo else 1)
            req_width = logical_width_bits
            
            # Physical primitives needed per bank
            # Ceiling division: needs enough blocks to cover depth and width
            blocks_depth = math.ceil(req_depth / BRAM_PHYSICAL_DEPTH)
            blocks_width = math.ceil(req_width / BRAM_PHYSICAL_WIDTH_BITS)
            
            per_bank_blocks = blocks_depth * blocks_width
            return per_bank_blocks * num_banks

        # 1. Input Buffer (Dense): width = 32 bits (4 INT8), Bank = Rows
        bram_in = get_blocks_for_buffer(hw_cfg.dense_buffer_depth, 
                                        hw_cfg.data_width_bits * 4, 
                                        hw_cfg.dense_buffer_bank, 
                                        hw_cfg.use_double_buffering)
        
        # 2. Weight Buffer (Sparse): width approx 8 bits (for calculation), Bank = Cols
        # Assuming compressed width fits efficiently
        bram_w = get_blocks_for_buffer(hw_cfg.sparse_buffer_depth, 
                                       hw_cfg.data_width_bits*2 + hw_cfg.mask_bits, 
                                       hw_cfg.sparse_buffer_bank, 
                                       hw_cfg.use_double_buffering)
        
        # 3. Output Buffer (Accumulators): width = 32 bits, Bank = Cols
        bram_out = get_blocks_for_buffer(hw_cfg.output_buffer_depth, 
                                         hw_cfg.accum_width_bits, 
                                         hw_cfg.output_buffer_bank, 
                                         False) # Output usually not PIPO in same way

        return bram_in + bram_w + bram_out

    def check_resource_constraints(self, hw_cfg: HardwareConfig) -> Tuple[bool, Dict[str, float], str]:
        # --- Resource Estimation ---
        lut_per_pe = 200   
        dsp_per_pe = 1     
        lut_per_bank_ctrl = 60 
        
        total_luts = (hw_cfg.pe_count * lut_per_pe) + \
                     ((hw_cfg.dense_buffer_bank + hw_cfg.sparse_buffer_bank + hw_cfg.output_buffer_bank) * lut_per_bank_ctrl)
        total_dsps = hw_cfg.pe_count * dsp_per_pe
        
        total_brams = self.calculate_bram_usage(hw_cfg)

        utils = {
            "lut": total_luts / ZCU102_LUT_TOTAL,
            "dsp": total_dsps / ZCU102_DSP_TOTAL,
            "bram": total_brams / ZCU102_BRAM_TOTAL
        }
        
        fail_reasons = []
        # Allow up to 100% physically, but Performance model will penalize high usage via Freq
        if utils["lut"] > 1.0: fail_reasons.append(f"LUT overflow ({utils['lut']:.2%})")
        if utils["dsp"] > 1.0: fail_reasons.append(f"DSP overflow ({utils['dsp']:.2%})")
        if utils["bram"] > 1.0: fail_reasons.append(f"BRAM overflow ({utils['bram']:.2%})")
        
        if fail_reasons:
            return False, utils, ", ".join(fail_reasons)
        return True, utils, "OK"

    def estimate_frequency(self, utils: Dict[str, float]) -> float:
        # Find the dominant resource constraint
        max_util = max(utils['lut'], utils['dsp'], utils['bram'])
        
        # Heuristic: Exponential decay penalty
        # If util is near 0, freq -> FREQ_BASE
        # If util is near 1.0, freq drops significantly
        penalty_factor = math.pow(max_util, 1.5) # Non-linear penalty
        current_freq = FREQ_BASE_MHZ * (1.0 - 0.6 * penalty_factor)
        
        return max(current_freq, FREQ_MIN_MHZ)

    def get_sram_energy_per_bit(self, total_size_bytes: int) -> float:
        size_kb = total_size_bytes / 1024.0
        if size_kb <= 1.0:
            return E_SRAM_BASE_PJ_BIT
        
        # Logarithmic scaling model
        scaling_factor = 1.0 + (math.log2(size_kb) * 0.15)
        return E_SRAM_BASE_PJ_BIT * scaling_factor

    def get_sparsity_compression_ratio(self, mode: SparsityMode, hw_cfg: HardwareConfig):
        if mode == SparsityMode.SPARSE_1_4: 
            return 0.25, (1*hw_cfg.data_width_bits + 3) / (4*hw_cfg.data_width_bits) 
        elif mode == SparsityMode.SPARSE_2_4: 
            return 0.5, (2*hw_cfg.data_width_bits + 3) / (4*hw_cfg.data_width_bits) 
        elif mode == SparsityMode.SPARSE_3_4: 
            return 0.75, (3*hw_cfg.data_width_bits + 3) / (4*hw_cfg.data_width_bits) 
        elif mode == SparsityMode.SPARSE_4_4: 
            return 1.0, 1.0
    
    def evaluate_workload(self, hw_cfg: HardwareConfig, wl: WorkloadConfig, utils: Dict[str, float]) -> PerfMetrics:
        real_freq_mhz = self.estimate_frequency(utils)
        
        M, K, N = wl.input_m, wl.input_k, wl.input_n
        Head = wl.input_head
        
        acc_ratio, comp_ratio = self.get_sparsity_compression_ratio(wl.sparse_mode, hw_cfg)
        
        num_tiles_m = math.ceil(M / hw_cfg.array_rows)
        num_tiles_n = math.ceil(N / hw_cfg.array_cols)
        total_tiles = num_tiles_m * num_tiles_n
        
        total_ops = M * N * K * Head
        effective_ops = total_ops * acc_ratio
        
        # Cycles Calculation using dynamic frequency for throughput, 
        # but pure cycle count remains logical.
        ideal_compute_cycles = math.ceil(effective_ops / hw_cfg.pe_count)
        systolic_pipeline_depth = hw_cfg.array_rows + hw_cfg.array_cols
        latency_overhead = systolic_pipeline_depth * total_tiles
        
        compute_cycles = ideal_compute_cycles + latency_overhead
        
        throughput_gops = (total_ops * 2 / (compute_cycles / real_freq_mhz)) / 1e3

        # Memory Traffic
        size_a_bytes = M * K * (hw_cfg.data_width_bits / 8)
        size_b_bytes = N * K * (hw_cfg.data_width_bits / 8) * comp_ratio
        size_c_bytes = M * N * (hw_cfg.data_width_bits / 8)
        
        traffic_mode_1 = size_a_bytes + (size_b_bytes * num_tiles_m)
        traffic_mode_2 = size_b_bytes + (size_a_bytes * num_tiles_n)
        dram_access_bytes = min(traffic_mode_1, traffic_mode_2) + size_c_bytes 
        dram_access_mb = dram_access_bytes / (1024 * 1024)
        
        bytes_per_cycle = (hw_cfg.dram_bw_gbps * 1e9) / (real_freq_mhz * 1e6)
        memory_cycles = math.ceil(dram_access_bytes / bytes_per_cycle)
        
        if memory_cycles > compute_cycles:
            total_cycles = memory_cycles
            bound_by = "Memory"
        else:
            total_cycles = compute_cycles
            bound_by = "Compute"
            
        # --- Energy Estimation ---
        
        # 1. DRAM Energy
        e_dram = (dram_access_bytes * 8) * E_DRAM_PJ_BIT
        
        # Calculate specific energy cost per bit based on buffer size
        e_sram_input_pj_bit = self.get_sram_energy_per_bit(hw_cfg.input_buffer_size_bytes)
        e_sram_weight_pj_bit = self.get_sram_energy_per_bit(hw_cfg.weight_buffer_size_bytes)
        
        # Approximate accesses based on traffic
        sram_access_bits = (traffic_mode_1 if traffic_mode_1 < traffic_mode_2 else traffic_mode_2) * 8
        # Weighted average for energy calculation (simplification)
        avg_sram_e_bit = (e_sram_input_pj_bit + e_sram_weight_pj_bit) / 2.0
        
        e_sram = (sram_access_bits * avg_sram_e_bit) + (size_c_bytes * 8 * e_sram_input_pj_bit)
        
        # 3. Compute Energy
        e_compute = effective_ops * E_MAC_PJ
        e_reg = total_ops * 3 * hw_cfg.data_width_bits * E_REG_PJ_BIT
        
        total_dynamic_energy_pj = e_dram + e_sram + e_compute + e_reg
        
        # Static power is consumed over the Real Time, which depends on Real Frequency
        execution_time_sec = total_cycles / (real_freq_mhz * 1e6)
        
        p_static_watts = P_STATIC_BASE_W + (utils['lut'] * P_STATIC_UTIL_FACTOR)
        e_static_pj = p_static_watts * execution_time_sec * 1e12
        
        total_energy_uj = (total_dynamic_energy_pj + e_static_pj) / 1e6
        
        return PerfMetrics(
            target_workload=wl,
            hw_config=hw_cfg,
            cycles_total=total_cycles,
            energy_uj=total_energy_uj,
            throughput_gops=throughput_gops,
            frequency_used_mhz=real_freq_mhz,
            dram_access_mb=dram_access_mb,
            bound_by=bound_by,
            utilization_resource=utils, 
            valid=True,
            reason="OK"
        )
    
def get_deit_small_workload() -> List[WorkloadConfig]:
    wls = []
    # Simplified workload generation for Deit-Small
    model_description = {
        "q_proj" : MatrixConfig(M=197, K=384, N=384),
        "matmul1": MatrixConfig(M=197, K=64, N=197, Head=6),
        "fc1" : MatrixConfig(M=197, K=1536, N=384),
        "fc2": MatrixConfig(M=197, K=384, N=1536),
    }
    
    for layer_name, desc in model_description.items():
        # Add diverse sparsity scenarios
        wls.append(WorkloadConfig(layer_name, desc.M, desc.N, desc.K, desc.Head, SparsityMode.SPARSE_2_4))
    
    return wls
  
def visualize_dse_results(results: List[Dict]):
    """
    Visualize DSE (Design Space Exploration) results with two plots:
    1. 3D surface plot of -log10(EDP) (higher is better)
    2. 2D contour/heatmap plot of EDP (lower is better)
    """
    if not results: 
        print("No results to visualize!")
        return

    # Extract data from results
    rows_data = [r['rows'] for r in results]
    cols_data = [r['cols'] for r in results]
    min_r, max_r = min(rows_data), max(rows_data)
    min_c, max_c = min(cols_data), max(cols_data)
    
    # Create grid for visualization
    ARRAY_SIZE_STEP = 4  # Assuming this is defined elsewhere
    unique_rows = np.arange(min_r, max_r + 1, ARRAY_SIZE_STEP)
    unique_cols = np.arange(min_c, max_c + 1, ARRAY_SIZE_STEP)
    X, Y = np.meshgrid(unique_cols, unique_rows)
    Z = np.full(X.shape, np.nan)  # For negative log EDP (3D plot)
    Z_edp = np.full(X.shape, np.nan)  # For raw EDP values (2D plot)
    
    # Create lookup dictionary for fast access
    res_lookup = {(r['rows'], r['cols']): r for r in results}
    
    # Fill the data matrices
    for i, r_val in enumerate(unique_rows):
        for j, c_val in enumerate(unique_cols):
            if (r_val, c_val) in res_lookup:
                rec = res_lookup[(r_val, c_val)]
                # Negative log for EDP (higher peak is better for visualization)
                if rec['edp'] > 0:
                    Z[i, j] = -np.log10(rec['edp'])
                    Z_edp[i, j] = rec['edp']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 7))
    
    # Plot 1: 3D EDP Surface (Negative log scale, higher is better)
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', 
                            linewidth=0.5, edgecolors='k', alpha=0.85)
    ax1.set_xlabel('Cols (N)')
    ax1.set_ylabel('Rows (M)')
    ax1.set_zlabel('-log10(EDP) [Higher is Better]')
    ax1.set_title('3D View: Energy-Delay Product (Higher Peak = Better EDP)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='-log10(EDP)')
    
    # Plot 2: 2D EDP Heatmap with Contours
    ax2 = fig.add_subplot(122)
    
    # Mask NaN values for better visualization
    Z_edp_masked = np.ma.array(Z_edp, mask=np.isnan(Z_edp))
    
    # Create contour plot with filled colors
    # Use log scale for better visualization of EDP range
    if np.nanmin(Z_edp) > 0:
        # Apply log scale for EDP visualization (since EDP values can vary widely)
        Z_edp_log = np.log10(Z_edp_masked)
        levels = np.linspace(np.nanmin(Z_edp_log), np.nanmax(Z_edp_log), 15)
        
        # Create filled contour plot
        contour = ax2.contourf(X, Y, Z_edp_log, levels=levels, 
                               cmap='plasma', alpha=0.8, extend='both')
        
        # Add contour lines
        contour_lines = ax2.contour(X, Y, Z_edp_log, levels=levels, 
                                    colors='k', linewidths=0.5, alpha=0.5)
        ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Set labels and title
        ax2.set_xlabel('Cols (N)')
        ax2.set_ylabel('Rows (M)')
        ax2.set_title('2D View: Energy-Delay Product (Lower is Better)\n(log10 scale)')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label('log10(EDP)')
        
        # Add markers for optimal points (lowest EDP)
        min_edp_idx = np.unravel_index(np.nanargmin(Z_edp), Z_edp.shape)
        ax2.plot(X[min_edp_idx], Y[min_edp_idx], 'r*', markersize=15, 
                 label=f'Best EDP: {Z_edp[min_edp_idx]:.2e}')
        ax2.legend(loc='upper right')
    else:
        # Fallback if EDP values are not positive
        ax2.text(0.5, 0.5, 'No valid EDP data for 2D plot', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('2D View: Energy-Delay Product')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

        
# def visualize_dse_results(results: List[Dict]):
#     if not results: return

#     rows_data = [r['rows'] for r in results]
#     cols_data = [r['cols'] for r in results]
#     min_r, max_r = min(rows_data), max(rows_data)
#     min_c, max_c = min(cols_data), max(cols_data)
    
#     unique_rows = np.arange(min_r, max_r + 1, ARRAY_SIZE_STEP)
#     unique_cols = np.arange(min_c, max_c + 1, ARRAY_SIZE_STEP)
#     X, Y = np.meshgrid(unique_cols, unique_rows)
#     Z = np.full(X.shape, np.nan)
#     F = np.full(X.shape, np.nan) # Frequency Map

#     res_lookup = {(r['rows'], r['cols']): r for r in results}
    
#     for i, r_val in enumerate(unique_rows):
#         for j, c_val in enumerate(unique_cols):
#             if (r_val, c_val) in res_lookup:
#                 rec = res_lookup[(r_val, c_val)]
#                 # Use negative log for EDP (lower is better, so higher peak is better)
#                 Z[i, j] = -np.log10(rec['edp']) if rec['edp'] > 0 else np.nan
#                 F[i, j] = rec['freq']

#     fig = plt.figure(figsize=(16, 7))
    
#     # Plot 1: EDP Surface
#     ax1 = fig.add_subplot(121, projection='3d')
#     surf = ax1.plot_surface(X, Y, Z, cmap='viridis', linewidth=0.5, edgecolors='k', alpha=0.85)
#     ax1.set_xlabel('Cols (N)')
#     ax1.set_ylabel('Rows (M)')
#     ax1.set_zlabel('-log10(EDP) [Higher is Better]')
#     ax1.set_title('Energy-Delay Product Efficiency')
    
#     # Plot 2: Frequency Heatmap
#     ax2 = fig.add_subplot(122)
#     im = ax2.imshow(F, extent=[min_c, max_c, min_r, max_r], origin='lower', cmap='plasma')
#     plt.colorbar(im, ax=ax2, label='Freq (MHz)')
#     ax2.set_xlabel('Cols (N)')
#     ax2.set_ylabel('Rows (M)')
#     ax2.set_title('Dynamic Frequency Scaling Result')
    
#     plt.tight_layout()
#     plt.show()

def run_dse_flexsparse_target_model(wls: List[WorkloadConfig]):
    dse = FlexSparseDSE()
    all_workloads = wls if isinstance(wls, list) else [wls]
    
    scan_range = list(range(ARRAY_SIZE_SRART, ARRAY_SIZE_END+1, ARRAY_SIZE_STEP)) 
    
    best_config = None
    best_edp = float('inf')
    dse_results = []
    
    print(f"{'RxC':<8} | {'Util(L/D/B)':<18} | {'Freq':<6} | {'Cycles':<10} | {'Energy':<10} | {'EDP':<10} | {'Bottle'}")
    print("-" * 90)
    
    for r in scan_range:
        for c in scan_range:
            hw_cfg = HardwareConfig(array_rows=r, array_cols=c)
            
            # 1. Check Resources (includes Improvement 4: BRAM check)
            valid_res, res_util, res_msg = dse.check_resource_constraints(hw_cfg)
            if not valid_res:
                continue

            total_cycles = 0
            total_energy = 0
            bottleneck_counts = {"Compute":0, "Memory":0}
            weighted_freq = 0
            
            # Evaluate workloads
            for wl in all_workloads:
                # Pass resource utilization to evaluate for Freq scaling (Improvement 1)
                metrics = dse.evaluate_workload(hw_cfg, wl, res_util)
                
                total_cycles += metrics.cycles_total
                total_energy += metrics.energy_uj
                bottleneck_counts[metrics.bound_by] += 1
                weighted_freq = metrics.frequency_used_mhz
            
            edp = total_energy * total_cycles
            dom_bottleneck = max(bottleneck_counts, key=bottleneck_counts.get)
            
            dse_results.append({
                'rows': r, 'cols': c,
                'edp': edp,
                'freq': weighted_freq,
                'cycles': total_cycles,
                'energy': total_energy
            })
            
            util_str = f"{res_util['lut']:.2f}/{res_util['dsp']:.2f}/{res_util['bram']:.2f}"
            print(f"{r:<3}x{c:<3} | {util_str:<18} | {int(weighted_freq):<6} | {total_cycles:<10} | {total_energy:<10.1f} | {edp:<10.2e} | {dom_bottleneck}")
                
            if edp < best_edp:
                best_edp = edp
                best_config = hw_cfg

    print("-" * 90)
    if best_config:
        print(f"\nOptimal: {best_config.array_rows} x {best_config.array_cols}")
        visualize_dse_results(dse_results)

