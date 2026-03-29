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
RESOURCE_BUDGET_RATIO = 0.8

BRAM_PHYSICAL_DEPTH = 1024
BRAM_PHYSICAL_WIDTH_BITS = 32

E_DRAM_PJ_BIT = 15.0
E_REG_PJ_BIT = 0.05
E_MAC_PJ = 0.2
E_SRAM_BASE_PJ_BIT = 0.1

P_STATIC_BASE_W = 3.5
P_STATIC_UTIL_FACTOR = 4.0

FREQ_BASE_MHZ = 350.0
FREQ_MIN_MHZ = 100.0

ARRAY_SIZE_START = 4
ARRAY_SIZE_END = 48
ARRAY_SIZE_STEP = 4

# --- Dataflow Enumeration ---
class Dataflow(Enum):
    INPUT_STATIONARY = 1
    WEIGHT_STATIONARY = 2
    OUTPUT_STATIONARY = 3

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
    dataflow: Dataflow = Dataflow.INPUT_STATIONARY  # NEW: dataflow type

    data_width_bits: int = 8   # INT8 for activations/weights
    accum_width_bits: int = 32 # internal accumulator width
    mask_bits: int = 3

    use_double_buffering: bool = True

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
        factor = 2 if self.use_double_buffering else 1
        return self.dense_buffer_depth * self.dense_buffer_bank * self.dense_bank_width_bytes * factor

    @property
    def weight_buffer_size_bytes(self):
        factor = 2 if self.use_double_buffering else 1
        # Rough estimation: 2x width for sparse metadata overhead
        return self.sparse_buffer_depth * self.sparse_buffer_bank * 2 * factor

    @property
    def output_buffer_size_bytes(self):
        # For Output Stationary, output is INT8 (quantized), else INT32
        if self.dataflow == Dataflow.OUTPUT_STATIONARY:
            out_width_bytes = self.data_width_bits // 8
        else:
            out_width_bytes = self.accum_width_bits // 8
        return self.output_buffer_depth * self.output_buffer_bank * out_width_bytes

@dataclass
class WorkloadConfig:
    layer_name: str
    input_m: int
    input_n: int
    input_k: int
    input_head: int
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
        """
        Calculates physical BRAM blocks needed based on Width and Depth requirements.
        Adapted for dataflow: output buffer width changes for output stationary.
        """
        def get_blocks_for_buffer(logical_depth, logical_width_bits, num_banks, is_pipo):
            req_depth = logical_depth * (2 if is_pipo else 1)
            req_width = logical_width_bits
            blocks_depth = math.ceil(req_depth / BRAM_PHYSICAL_DEPTH)
            blocks_width = math.ceil(req_width / BRAM_PHYSICAL_WIDTH_BITS)
            return blocks_depth * blocks_width * num_banks

        # Input buffer: always width = 4 * data_width (INT8)
        bram_in = get_blocks_for_buffer(hw_cfg.dense_buffer_depth,
                                        4 * hw_cfg.data_width_bits,
                                        hw_cfg.dense_buffer_bank,
                                        hw_cfg.use_double_buffering)

        # Weight buffer: roughly 2*data_width + mask bits
        bram_w = get_blocks_for_buffer(hw_cfg.sparse_buffer_depth,
                                       2 * hw_cfg.data_width_bits + hw_cfg.mask_bits,
                                       hw_cfg.sparse_buffer_bank,
                                       hw_cfg.use_double_buffering)

        # Output buffer: width depends on dataflow
        out_width_bits = (hw_cfg.accum_width_bits if hw_cfg.dataflow != Dataflow.OUTPUT_STATIONARY
                          else hw_cfg.data_width_bits)
        bram_out = get_blocks_for_buffer(hw_cfg.output_buffer_depth,
                                         out_width_bits,
                                         hw_cfg.output_buffer_bank,
                                         False)

        return bram_in + bram_w + bram_out

    def check_resource_constraints(self, hw_cfg: HardwareConfig) -> Tuple[bool, Dict[str, float], str]:
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
        if utils["lut"] > 1.0: fail_reasons.append(f"LUT overflow ({utils['lut']:.2%})")
        if utils["dsp"] > 1.0: fail_reasons.append(f"DSP overflow ({utils['dsp']:.2%})")
        if utils["bram"] > 1.0: fail_reasons.append(f"BRAM overflow ({utils['bram']:.2%})")

        if fail_reasons:
            return False, utils, ", ".join(fail_reasons)
        return True, utils, "OK"

    def estimate_frequency(self, utils: Dict[str, float]) -> float:
        max_util = max(utils['lut'], utils['dsp'], utils['bram'])
        penalty_factor = math.pow(max_util, 1.5)
        current_freq = FREQ_BASE_MHZ * (1.0 - 0.6 * penalty_factor)
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
        elif mode == SparsityMode.SPARSE_4_4:
            return 1.0, 1.0

    def evaluate_workload(self, hw_cfg: HardwareConfig, wl: WorkloadConfig, utils: Dict[str, float]) -> PerfMetrics:
        real_freq_mhz = self.estimate_frequency(utils)

        M, K, N = wl.input_m, wl.input_k, wl.input_n
        Head = wl.input_head

        acc_ratio, comp_ratio = self.get_sparsity_compression_ratio(wl.sparse_mode, hw_cfg)

        # Tiling dimensions based on array size
        num_tiles_m = math.ceil(M / hw_cfg.array_rows)
        num_tiles_n = math.ceil(N / hw_cfg.array_cols)
        total_tiles = num_tiles_m * num_tiles_n

        total_ops = M * N * K * Head
        effective_ops = total_ops * acc_ratio

        # Compute cycles (ideal + pipeline overhead)
        ideal_compute_cycles = math.ceil(effective_ops / hw_cfg.pe_count)
        systolic_pipeline_depth = hw_cfg.array_rows + hw_cfg.array_cols
        latency_overhead = systolic_pipeline_depth * total_tiles
        compute_cycles = ideal_compute_cycles + latency_overhead

        # Memory traffic to/from DRAM
        size_a_bytes = M * K * (hw_cfg.data_width_bits // 8)
        size_b_bytes = N * K * (hw_cfg.data_width_bits // 8) * comp_ratio
        # Output size depends on dataflow (INT8 for output stationary, else INT32)
        out_size_bytes = (M * N * (hw_cfg.data_width_bits // 8)
                          if hw_cfg.dataflow == Dataflow.OUTPUT_STATIONARY
                          else M * N * (hw_cfg.accum_width_bits // 8))

        # DRAM traffic: two tiling schemes (minimizing DRAM reads)
        traffic_mode_1 = size_a_bytes + (size_b_bytes * num_tiles_m)
        traffic_mode_2 = size_b_bytes + (size_a_bytes * num_tiles_n)
        dram_access_bytes = min(traffic_mode_1, traffic_mode_2) + out_size_bytes
        dram_access_mb = dram_access_bytes / (1024 * 1024)

        # Memory cycles
        bytes_per_cycle = (hw_cfg.dram_bw_gbps * 1e9) / (real_freq_mhz * 1e6)
        memory_cycles = math.ceil(dram_access_bytes / bytes_per_cycle)

        # Determine bottleneck
        if memory_cycles > compute_cycles:
            total_cycles = memory_cycles
            bound_by = "Memory"
        else:
            total_cycles = compute_cycles
            bound_by = "Compute"

        # --- Energy Estimation ---
        # DRAM energy
        e_dram = (dram_access_bytes * 8) * E_DRAM_PJ_BIT

        # SRAM energy per bit based on buffer sizes
        e_sram_input_pj_bit = self.get_sram_energy_per_bit(hw_cfg.input_buffer_size_bytes)
        e_sram_weight_pj_bit = self.get_sram_energy_per_bit(hw_cfg.weight_buffer_size_bytes)

        # --- Dataflow‑dependent on‑chip data movement ---
        # We approximate total SRAM access bits as the sum of:
        # - Input tile reads (number of times input tile is fetched from input buffer to PEs)
        # - Weight tile reads
        # - Output tile writes/reads
        # The exact numbers depend on reuse pattern. Here we use simple multipliers:
        if hw_cfg.dataflow == Dataflow.INPUT_STATIONARY:
            # Input reused across columns, weight reused across rows
            input_read_factor = num_tiles_m * num_tiles_n   # each input tile read once per output tile
            weight_read_factor = num_tiles_m                 # each weight tile reused across rows
            output_write_factor = 1                         # output written once per tile
        elif hw_cfg.dataflow == Dataflow.WEIGHT_STATIONARY:
            # Weight reused across rows, input reused across columns
            input_read_factor = num_tiles_n
            weight_read_factor = num_tiles_m * num_tiles_n
            output_write_factor = 1
        else:  # OUTPUT_STATIONARY
            # Output partial sums accumulate on chip; input and weight are streamed
            # Each input tile read once, each weight tile read once, output read/written multiple times
            input_read_factor = num_tiles_m * num_tiles_n
            weight_read_factor = num_tiles_m * num_tiles_n
            # Output is written once per tile (final accumulation)
            output_write_factor = 1

        # SRAM access bits for inputs, weights, outputs
        sram_input_bits = size_a_bytes * 8 * input_read_factor
        sram_weight_bits = size_b_bytes * 8 * weight_read_factor
        sram_output_bits = out_size_bytes * 8 * output_write_factor

        # Combine SRAM energies (use average of input and weight SRAM energy for simplicity)
        avg_sram_e_bit = (e_sram_input_pj_bit + e_sram_weight_pj_bit) / 2.0
        e_sram = (sram_input_bits + sram_weight_bits) * avg_sram_e_bit + sram_output_bits * e_sram_input_pj_bit

        # Compute energy
        e_compute = effective_ops * E_MAC_PJ
        e_reg = total_ops * 3 * hw_cfg.data_width_bits * E_REG_PJ_BIT

        total_dynamic_energy_pj = e_dram + e_sram + e_compute + e_reg

        # Static energy
        execution_time_sec = total_cycles / (real_freq_mhz * 1e6)
        p_static_watts = P_STATIC_BASE_W + (utils['lut'] * P_STATIC_UTIL_FACTOR)
        e_static_pj = p_static_watts * execution_time_sec * 1e12

        total_energy_uj = (total_dynamic_energy_pj + e_static_pj) / 1e6

        # Throughput (GOPS)
        throughput_gops = (total_ops * 2 / (total_cycles / real_freq_mhz)) / 1e3

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
        "q_proj": MatrixConfig(M=197, K=384, N=384),
        "matmul1": MatrixConfig(M=197, K=64, N=197, Head=6),
        "fc1": MatrixConfig(M=197, K=1536, N=384),
        "fc2": MatrixConfig(M=197, K=384, N=1536),
    }

    for layer_name, desc in model_description.items():
        wls.append(WorkloadConfig(layer_name, desc.M, desc.N, desc.K, desc.Head, SparsityMode.SPARSE_2_4))
    return wls

def visualize_dse_results(results: List[Dict]):
    """
    Visualize DSE results with two plots:
    1. 3D surface of -log10(EDP) (higher is better)
    2. 2D contour/heatmap of EDP (lower is better)
    """
    if not results:
        print("No results to visualize!")
        return

    rows_data = [r['rows'] for r in results]
    cols_data = [r['cols'] for r in results]
    min_r, max_r = min(rows_data), max(rows_data)
    min_c, max_c = min(cols_data), max(cols_data)

    unique_rows = np.arange(min_r, max_r + 1, ARRAY_SIZE_STEP)
    unique_cols = np.arange(min_c, max_c + 1, ARRAY_SIZE_STEP)
    X, Y = np.meshgrid(unique_cols, unique_rows)
    Z = np.full(X.shape, np.nan)
    Z_edp = np.full(X.shape, np.nan)

    res_lookup = {(r['rows'], r['cols']): r for r in results}

    for i, r_val in enumerate(unique_rows):
        for j, c_val in enumerate(unique_cols):
            if (r_val, c_val) in res_lookup:
                rec = res_lookup[(r_val, c_val)]
                if rec['edp'] > 0:
                    Z[i, j] = -np.log10(rec['edp'])
                    Z_edp[i, j] = rec['edp']

    fig = plt.figure(figsize=(16, 7))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', linewidth=0.5, edgecolors='k', alpha=0.85)
    ax1.set_xlabel('Cols (N)')
    ax1.set_ylabel('Rows (M)')
    ax1.set_zlabel('-log10(EDP) [Higher is Better]')
    ax1.set_title('3D View: Energy-Delay Product')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='-log10(EDP)')

    # 2D heatmap
    ax2 = fig.add_subplot(122)
    Z_edp_masked = np.ma.array(Z_edp, mask=np.isnan(Z_edp))
    if np.nanmin(Z_edp) > 0:
        Z_edp_log = np.log10(Z_edp_masked)
        levels = np.linspace(np.nanmin(Z_edp_log), np.nanmax(Z_edp_log), 15)
        contour = ax2.contourf(X, Y, Z_edp_log, levels=levels, cmap='plasma', alpha=0.8, extend='both')
        contour_lines = ax2.contour(X, Y, Z_edp_log, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
        ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax2.set_xlabel('Cols (N)')
        ax2.set_ylabel('Rows (M)')
        ax2.set_title('2D View: Energy-Delay Product (log10 scale)')
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label('log10(EDP)')
        min_edp_idx = np.unravel_index(np.nanargmin(Z_edp), Z_edp.shape)
        ax2.plot(X[min_edp_idx], Y[min_edp_idx], 'r*', markersize=15, label=f'Best EDP: {Z_edp[min_edp_idx]:.2e}')
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No valid EDP data for 2D plot', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('2D View: Energy-Delay Product')

    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def run_dse_flexsparse_target_model(wls: List[WorkloadConfig], dataflow: Dataflow):
    """
    Run DSE for a specific dataflow type.
    """
    dse = FlexSparseDSE()
    all_workloads = wls if isinstance(wls, list) else [wls]

    scan_range = list(range(ARRAY_SIZE_START, ARRAY_SIZE_END + 1, ARRAY_SIZE_STEP))

    best_config = None
    best_edp = float('inf')
    dse_results = []

    print(f"\n=== Dataflow: {dataflow.name} ===")
    print(f"{'RxC':<8} | {'Util(L/D/B)':<18} | {'Freq':<6} | {'Cycles':<10} | {'Energy':<10} | {'EDP':<10} | {'Bottle'}")
    print("-" * 90)

    for r in scan_range:
        for c in scan_range:
            hw_cfg = HardwareConfig(array_rows=r, array_cols=c, dataflow=dataflow)

            valid_res, res_util, res_msg = dse.check_resource_constraints(hw_cfg)
            if not valid_res:
                continue

            total_cycles = 0
            total_energy = 0
            bottleneck_counts = {"Compute": 0, "Memory": 0}
            weighted_freq = 0

            for wl in all_workloads:
                metrics = dse.evaluate_workload(hw_cfg, wl, res_util)

                total_cycles += metrics.cycles_total
                total_energy += metrics.energy_uj
                bottleneck_counts[metrics.bound_by] += 1
                weighted_freq = metrics.frequency_used_mhz  # average frequency (similar for all layers)

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
        print(f"Optimal for {dataflow.name}: {best_config.array_rows} x {best_config.array_cols}, EDP = {best_edp:.2e}")
        visualize_dse_results(dse_results)
    return dse_results, best_config, best_edp

if __name__ == "__main__":
    workloads = get_deit_small_workload()

    # Evaluate for each dataflow
    for df in [Dataflow.INPUT_STATIONARY, Dataflow.WEIGHT_STATIONARY, Dataflow.OUTPUT_STATIONARY]:
        results, best_cfg, best_edp = run_dse_flexsparse_target_model(workloads, df)