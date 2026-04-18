import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Hardware Constants ---
ZCU102_LUT_TOTAL = 274080
ZCU102_DSP_TOTAL = 2520
ZCU102_BRAM_TOTAL = 912  # 36Kb blocks
RESOURCE_BUDGET = 0.6

# BRAM physical organization
BRAM_PHYSICAL_DEPTH = 512
BRAM_PHYSICAL_WIDTH_BITS = 32 

# Energy Estimation 
E_MAC_PJ = 0.3          
E_DRAM_PJ_BIT = 20.0    
E_REG_PJ_BIT = 0.2     
E_SRAM_BASE_PJ_BIT = 0.4  

P_STATIC_BASE_W = 1.0   
P_STATIC_UTIL_FACTOR = 3.5  

FREQ_BASE_MHZ = 400.0   
FREQ_MIN_MHZ = 100.0
FREQ_PENALTY_COEFF = 0.1    

# Search space (Expanded to include Banks for 3D exploration)
ARRAY_BANK_START = 2
ARRAY_BANK_END = 16
ARRAY_BANK_STEP = 2

ARRAY_SIZE_START = 4
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
    # 3D Extension: Added array_banks to define the number of 2D planes
    array_banks: int      
    array_rows: int       
    array_cols: int
    
    data_width_bits: int = 8
    accum_width_bits: int = 32
    mask_bits: int = 4
    
    use_double_buffering: bool = True 
    
    # Reduced buffer depths to limit BRAM usage for large arrays
    dense_buffer_depth: int = 512
    sparse_buffer_depth: int = 512
    output_buffer_depth: int = 512
    
    dram_bw_gbps = 3.6
    
    def __post_init__(self):
        # 3D Extension: Scale buffer banks linearly with the number of processing banks
        self.dense_buffer_bank = self.array_banks * self.array_rows
        self.sparse_buffer_bank = self.array_banks * self.array_cols
        self.output_buffer_bank = self.array_banks * self.array_cols
    
    @property   
    def pe_count(self):
        # 3D Extension: Total PE count is Banks * Rows * Cols
        return self.array_banks * self.array_rows * self.array_cols

    @property
    def dense_bank_width_bytes(self):
        return 4 * (self.data_width_bits // 8) 

    @property
    def input_buffer_size_bytes(self):
        factor = 2 if self.use_double_buffering else 1
        return self.dense_buffer_depth * self.dense_buffer_bank * self.dense_bank_width_bytes * factor
    
    @property
    def weight_buffer_size_bytes(self):
        factor = 2 if self.use_double_buffering else 1
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
    frequency_used_mhz: float = 0.0
    dram_access_mb: float = 0.0
    
    bound_by: str = "Compute"
    utilization_resource: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""
    
class FlexSparseDSE:
    def __init__(self):
        pass
    
    def calculate_bram_usage(self, hw_cfg: HardwareConfig) -> int:
        """Calculate physical BRAM blocks (36Kb each) required"""
        def get_blocks_for_buffer(logical_depth, logical_width_bits, num_banks, is_pipo):
            req_depth = logical_depth * (2 if is_pipo else 1)
            req_width = logical_width_bits
            blocks_depth = math.ceil(req_depth / BRAM_PHYSICAL_DEPTH)
            blocks_width = math.ceil(req_width / BRAM_PHYSICAL_WIDTH_BITS)
            per_bank_blocks = blocks_depth * blocks_width
            return per_bank_blocks * num_banks

        bram_in = get_blocks_for_buffer(hw_cfg.dense_buffer_depth, 
                                        hw_cfg.data_width_bits * 4, 
                                        hw_cfg.dense_buffer_bank, 
                                        hw_cfg.use_double_buffering)
        bram_w = get_blocks_for_buffer(hw_cfg.sparse_buffer_depth, 
                                       hw_cfg.data_width_bits*2 + hw_cfg.mask_bits, 
                                       hw_cfg.sparse_buffer_bank, 
                                       hw_cfg.use_double_buffering)
        bram_out = get_blocks_for_buffer(hw_cfg.output_buffer_depth, 
                                         hw_cfg.accum_width_bits, 
                                         hw_cfg.output_buffer_bank, 
                                         False)
        return bram_in + bram_w + bram_out

    def check_resource_constraints(self, hw_cfg: HardwareConfig) -> Tuple[bool, Dict[str, float], str]:
        lut_per_pe = 50
        dsp_per_pe = 1                     # INT8 multiplier without DSP
        lut_per_bank_ctrl = 200 
        
        # Total logic naturally scales up with 3D PE count and 3D Bank controllers
        total_luts = (hw_cfg.pe_count * lut_per_pe) + \
                     ((hw_cfg.dense_buffer_bank + hw_cfg.sparse_buffer_bank + hw_cfg.output_buffer_bank) * lut_per_bank_ctrl)
        total_dsps = hw_cfg.pe_count * dsp_per_pe
        total_brams = self.calculate_bram_usage(hw_cfg)

        utils = {
            "lut": total_luts / (ZCU102_LUT_TOTAL * RESOURCE_BUDGET),
            "dsp": total_dsps / (ZCU102_DSP_TOTAL),
            "bram": total_brams / (ZCU102_BRAM_TOTAL * RESOURCE_BUDGET)
        }
        
        fail_reasons = []
        if utils["lut"] > 1.0: fail_reasons.append(f"LUT overflow ({utils['lut']:.2%})")
        if utils["dsp"] > 1.0: fail_reasons.append(f"DSP overflow ({utils['dsp']:.2%})")
        if utils["bram"] > 1.0: fail_reasons.append(f"BRAM overflow ({utils['bram']:.2%})")
        
        if fail_reasons:
            return False, utils, ", ".join(fail_reasons)
        return True, utils, "OK"

    def estimate_frequency(self, utils: Dict[str, float]) -> float:
        """Frequency scales down with max resource utilization (stronger penalty)"""
        max_util = max(utils['lut'], utils['dsp'], utils['bram'])
        penalty_factor = math.pow(max_util, 1.5)
        current_freq = FREQ_BASE_MHZ * (1.0 - FREQ_PENALTY_COEFF * penalty_factor)
        return max(current_freq, FREQ_MIN_MHZ)

    def get_sram_energy_per_bit(self, total_size_bytes: int) -> float:
        size_kb = total_size_bytes / 1024.0
        if size_kb <= 1.0:
            return E_SRAM_BASE_PJ_BIT
        scaling_factor = 1.0 + (math.log2(size_kb) * 0.15)
        return E_SRAM_BASE_PJ_BIT * scaling_factor

    def get_sparsity_compression_ratio(self, mode: SparsityMode, hw_cfg: HardwareConfig):
        if mode == SparsityMode.SPARSE_1_4: 
            return 0.25, (1*hw_cfg.data_width_bits + 3) / (4*hw_cfg.data_width_bits) 
        elif mode == SparsityMode.SPARSE_2_4: 
            return 0.5, (2*hw_cfg.data_width_bits + 3) / (4*hw_cfg.data_width_bits) 
        elif mode == SparsityMode.SPARSE_3_4: 
            return 0.75, (3*hw_cfg.data_width_bits + 3) / (4*hw_cfg.data_width_bits) 
        else:  # SPARSE_4_4
            return 1.0, 1.0
    
    def evaluate_workload(self, hw_cfg: HardwareConfig, wl: WorkloadConfig, utils: Dict[str, float]) -> PerfMetrics:
        real_freq_mhz = self.estimate_frequency(utils)
        
        M, K, N = wl.input_m, wl.input_k, wl.input_n
        Head = wl.input_head
        
        # Merge Head into row dimension for tiling (activation matrix: (M*Head) x K)
        total_rows = M * Head
        
        acc_ratio, comp_ratio = self.get_sparsity_compression_ratio(wl.sparse_mode, hw_cfg)
        
        # 3D Extension: Distribute total_rows across multiple banks spatially
        # Each bank group processes a contiguous set of rows; total parallel rows = array_banks * array_rows
        parallel_rows = hw_cfg.array_banks * hw_cfg.array_rows
        parallel_cols = hw_cfg.array_cols
        
        num_tiles_m = math.ceil(total_rows / parallel_rows)   # tiles along row dimension
        num_tiles_n = math.ceil(N / parallel_cols)            # tiles along column dimension
        total_tiles = num_tiles_m * num_tiles_n
        
        total_ops = total_rows * N * K   # Head already accounted in total_rows
        effective_ops = total_ops * acc_ratio
        
        ideal_compute_cycles = math.ceil(effective_ops / hw_cfg.pe_count)
        systolic_pipeline_depth = hw_cfg.array_rows + hw_cfg.array_cols
        latency_overhead = systolic_pipeline_depth * wl.sparse_mode.value * total_tiles
        compute_cycles = ideal_compute_cycles + latency_overhead
        
        throughput_gops = (total_ops * 2 / (compute_cycles / real_freq_mhz)) / 1e3

        # --- Memory traffic with proper data reuse modeling ---
        # Activation (dense) size: total_rows * K * (data_width_bits/8)
        size_a_bytes = total_rows * K * (hw_cfg.data_width_bits / 8)
        # Weight (sparse) size: K * N * (data_width_bits/8) * comp_ratio (compressed)
        size_b_bytes = K * N * (hw_cfg.data_width_bits / 8) * comp_ratio
        # Output (dense) size: total_rows * N * (data_width_bits/8)
        size_c_bytes = total_rows * N * (hw_cfg.data_width_bits / 8)
        
        # Two tiling strategies:
        # 1) Tile along M (row) dimension: each tile loads a different activation chunk,
        #    but the same weight matrix can be reused across all M tiles.
        traffic_mode_1 = (size_a_bytes * num_tiles_m) + size_b_bytes   # A repeated, B once
        # 2) Tile along N (col) dimension: each tile loads a different weight chunk,
        #    but the same activation matrix can be reused across all N tiles.
        traffic_mode_2 = size_a_bytes + (size_b_bytes * num_tiles_n)   # A once, B repeated
        
        # Choose the strategy that minimizes DRAM traffic
        dram_access_bytes = min(traffic_mode_1, traffic_mode_2) + size_c_bytes   # + output write-back
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
        e_dram = (dram_access_bytes * 8) * E_DRAM_PJ_BIT
        
        e_sram_input_pj_bit = self.get_sram_energy_per_bit(hw_cfg.input_buffer_size_bytes)
        e_sram_weight_pj_bit = self.get_sram_energy_per_bit(hw_cfg.weight_buffer_size_bytes)
        # Total SRAM access bits: the smaller of the two traffic modes (excluding output write)
        sram_access_bits = min(traffic_mode_1, traffic_mode_2) * 8
        avg_sram_e_bit = (e_sram_input_pj_bit + e_sram_weight_pj_bit) / 2.0
        e_sram = (sram_access_bits * avg_sram_e_bit) + (size_c_bytes * 8 * e_sram_input_pj_bit)
        
        e_compute = effective_ops * E_MAC_PJ
        e_reg = total_ops * 3 * hw_cfg.data_width_bits * E_REG_PJ_BIT
        
        total_dynamic_energy_pj = e_dram + e_sram + e_compute + e_reg
        
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
    model_description = {
        "qkv_proj" : MatrixConfig(M=197, K=384, N=1152),
        "o_proj" : MatrixConfig(M=197, K=384, N=384),
        "matmul1": MatrixConfig(M=197, K=64, N=197, Head=6),
        "matmul2": MatrixConfig(M=197, K=64, N=197, Head=6),
        "fc1" : MatrixConfig(M=197, K=1536, N=384),
        "fc2": MatrixConfig(M=197, K=384, N=1536),
    }
    for layer_name, desc in model_description.items():
        wls.append(WorkloadConfig(layer_name, desc.M, desc.N, desc.K, desc.Head, SparsityMode.SPARSE_2_4))
    return wls
  
def visualize_dse_results(results: List[Dict], best_bank: int):
    """
    Visualize DSE results:
    - 3D scatter plot: all bank configurations, color-coded by bank number.
    - 2D contour plot: results for a specific bank (best_bank) to show EDP landscape.
    """
    if not results:
        print("No results to visualize!")
        return

    # Prepare data for 3D scatter (all banks)
    banks = []
    rows = []
    cols = []
    edp_vals = []
    log_edp_vals = []   # -log10(EDP) for better visualization (higher = better)

    for r in results:
        b = r['banks']
        rw = r['rows']
        cl = r['cols']
        edp = r['edp']
        if edp > 0:                     # avoid log(<=0)
            banks.append(b)
            rows.append(rw)
            cols.append(cl)
            edp_vals.append(edp)
            log_edp_vals.append(-np.log10(edp))

    if not log_edp_vals:
        print("No valid EDP values (all zero or negative).")
        return

    # --- 3D Scatter Plot: all banks, color-coded ---
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')

    # Use a colormap to map bank numbers to colors
    unique_banks = sorted(set(banks))
    cmap = plt.cm.get_cmap('tab10', len(unique_banks))  # discrete colormap
    for bank in unique_banks:
        mask = [b == bank for b in banks]
        bank_rows = [rows[i] for i, m in enumerate(mask) if m]
        bank_cols = [cols[i] for i, m in enumerate(mask) if m]
        bank_log_edp = [log_edp_vals[i] for i, m in enumerate(mask) if m]
        ax1.scatter(bank_cols, bank_rows, bank_log_edp,
                    label=f'Banks = {bank}',
                    c=[cmap(unique_banks.index(bank))],
                    marker='o', s=40, alpha=0.8, edgecolors='k', linewidth=0.5)

    ax1.set_xlabel('Cols (N)', fontsize=10)
    ax1.set_ylabel('Rows (M)', fontsize=10)
    ax1.set_zlabel('-log10(EDP) [Higher is Better]', fontsize=10)
    ax1.set_title('3D Exploration Space: All Bank Configurations\n(Color = Bank Number)', fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=8, framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # --- 2D Contour Plot: only the best bank (or specified bank) ---
    results_slice = [r for r in results if r['banks'] == best_bank]
    if not results_slice:
        print(f"No results found for bank = {best_bank}, skipping 2D contour.")
    else:
        ax2 = fig.add_subplot(122)

        # Build grid data for the selected bank
        unique_rows = sorted(set(r['rows'] for r in results_slice))
        unique_cols = sorted(set(r['cols'] for r in results_slice))
        X, Y = np.meshgrid(unique_cols, unique_rows)
        Z_edp = np.full(X.shape, np.nan)

        res_lookup = {(r['rows'], r['cols']): r['edp'] for r in results_slice}
        for i, r_val in enumerate(unique_rows):
            for j, c_val in enumerate(unique_cols):
                if (r_val, c_val) in res_lookup:
                    Z_edp[i, j] = res_lookup[(r_val, c_val)]

        # Mask invalid entries
        Z_masked = np.ma.masked_invalid(Z_edp)
        if Z_masked.count() > 0:
            # Use log10(EDP) for contour levels (lower is better)
            log_edp = np.log10(Z_masked)
            levels = np.linspace(np.nanmin(log_edp), np.nanmax(log_edp), 15)
            contourf = ax2.contourf(X, Y, log_edp, levels=levels, cmap='plasma', alpha=0.8)
            contour = ax2.contour(X, Y, log_edp, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
            ax2.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
            cbar = plt.colorbar(contourf, ax=ax2)
            cbar.set_label('log10(EDP) [Lower is Better]', fontsize=9)

            # Mark the best EDP point in this bank
            min_idx = np.unravel_index(np.nanargmin(Z_edp), Z_edp.shape)
            ax2.plot(X[min_idx], Y[min_idx], 'r*', markersize=15,
                     label=f'Best EDP: {Z_edp[min_idx]:.2e}')
            ax2.legend(loc='upper right', fontsize=9)

        else:
            ax2.text(0.5, 0.5, 'No valid EDP data for 2D plot',
                     ha='center', va='center', transform=ax2.transAxes)

        ax2.set_xlabel('Cols (N)', fontsize=10)
        ax2.set_ylabel('Rows (M)', fontsize=10)
        ax2.set_title(f'2D Contour: Energy-Delay Product (Banks = {best_bank})', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def run_dse_flexsparse_target_model(wls: List[WorkloadConfig]):
    dse = FlexSparseDSE()
    all_workloads = wls if isinstance(wls, list) else [wls]
    
    # 3D Extension: Added bank_range for the extra nested loop
    bank_range = list(range(ARRAY_BANK_START, ARRAY_BANK_END+1, ARRAY_BANK_STEP))
    scan_range = list(range(ARRAY_SIZE_START, ARRAY_SIZE_END+1, ARRAY_SIZE_STEP)) 
    
    best_config = None
    best_edp = float('inf')
    dse_results = []
    
    print(f"{'BxRxC':<10} | {'Util(L/D/B)':<18} | {'Freq':<6} | {'Cycles':<10} | {'Energy':<10} | {'EDP':<10} | {'Bottle'}")
    print("-" * 92)
    
    # 3D Extension: Outer loop sweeps through array banks
    for b in bank_range:
        for r in scan_range:
            for c in scan_range:
                hw_cfg = HardwareConfig(array_banks=b, array_rows=r, array_cols=c)
                _, res_util, _ = dse.check_resource_constraints(hw_cfg)
                
                # Resource Constraint Checking
                is_valid, res_util, msg = dse.check_resource_constraints(hw_cfg)
                
                if not is_valid:
                    # Skip invalid configurations to ensure the solution is feasible
                    continue

                total_cycles = 0
                total_energy = 0
                bottleneck_counts = {"Compute":0, "Memory":0}
                weighted_freq = 0
                
                for wl in all_workloads:
                    metrics = dse.evaluate_workload(hw_cfg, wl, res_util)
                    total_cycles += metrics.cycles_total
                    total_energy += metrics.energy_uj
                    bottleneck_counts[metrics.bound_by] += 1
                    weighted_freq = metrics.frequency_used_mhz
                
                edp = total_energy * total_cycles
                dom_bottleneck = max(bottleneck_counts, key=bottleneck_counts.get)
                
                dse_results.append({
                    'banks': b, 'rows': r, 'cols': c,
                    'edp': edp,
                    'freq': weighted_freq,
                    'cycles': total_cycles,
                    'energy': total_energy,
                })
                
                util_str = f"{res_util['lut']:.2f}/{res_util['dsp']:.2f}/{res_util['bram']:.2f}"
                dims_str = f"{b}x{r}x{c}"
                print(f"{dims_str:<10} | {util_str:<18} | {int(weighted_freq):<6} | {total_cycles:<10} | {total_energy:<10.1f} | {edp:<10.2e} | {dom_bottleneck}")
                    
                if edp < best_edp :
                    best_edp = edp
                    best_config = hw_cfg

    print("-" * 92)
    if best_config:
        print(f"\nOptimal 3D PE array: [Banks={best_config.array_banks}, Rows={best_config.array_rows}, Cols={best_config.array_cols}]")
        visualize_dse_results(dse_results, best_bank=best_config.array_banks)
        # plot_roofline_relationship(dse_results, all_workloads, dse)

if __name__ == "__main__":
    workloads = get_deit_small_workload()
    run_dse_flexsparse_target_model(workloads)