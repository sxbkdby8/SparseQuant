import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import math

VERBOSE_ = True

def print_verbose(msg, *args):
    if VERBOSE_:
        print(msg, *args)
        
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.floor(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output

round_ste = RoundSTE.apply
floor_ste = FloorSTE.apply

class IntSoftmaxFixed(nn.Module):
    def __init__(self, M: int = 4, N: int = 8, LN_SUM_DW: int = 16):
        super().__init__()
        # self.n_bits = n_bits
        self.M = M
        self.N = N
        
        # 存储缩放因子用于反向传播
        self.register_buffer('input_scale', torch.tensor(1.0 / (2**N)))
        self.register_buffer('exp_scale', torch.tensor(1.0))
        self.register_buffer('ln_sum_scale', torch.tensor(1.0 / (2**N)))
        
    class IntExpSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_float, M, N):
            ctx.save_for_backward(x_float)
            ctx.M = M
            ctx.N = N
            
            scale = 1.0 / (2**N)
            x_int = torch.round(x_float / scale)
            x_int = torch.clamp(x_int, -2**(M+N-1), 2**(M+N-1)-1)
            x_poly = x_int + x_int / 2 - x_int / 16
            exp_approx = 2.0 ** (x_poly / (2**N))
            exp_int = torch.round(exp_approx / scale)
            exp_int = torch.clamp(exp_int, 0, 2**(M+N)-1)
            
            return exp_int * scale  # 返回浮点数但保持定点数语义
        
        @staticmethod
        def backward(ctx, grad_output):
            x_float, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input = torch.clamp(grad_input, -1.0, 1.0)
            return grad_input, None, None
    
    class IntLnSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_float, M, N):
            ctx.save_for_backward(x_float)
            ctx.M = M
            ctx.N = N
            
            scale = 1.0 / (2**N)
            x_int = torch.round(x_float / scale)
            x_int = torch.clamp(x_int, 1, 2**(M+N)-1)  # ln(0)未定义
            
            lod = torch.floor(torch.log2(x_int.float()))
            dec = (x_int / (2**lod) - 1) * (2**N)
            dec = torch.floor(dec) / (2**N)
            rst = dec + lod - N
            
            ln_approx = rst/2 + rst/4 - rst/16
            
            ln_int = torch.round(ln_approx / scale)
            return ln_int * scale
        
        @staticmethod
        def backward(ctx, grad_output):
            x_float, = ctx.saved_tensors
            grad_input = grad_output.clone() / (x_float + 1e-8)  # d(ln(x))/dx = 1/x
            grad_input = torch.clamp(grad_input, -1.0, 1.0)
            return grad_input, None, None
    
    def forward(self, x: torch.Tensor):
        x_float = x
        
        x_max = x_float.max(dim=-1, keepdim=True).values
        x_shifted = x_float - x_max
        
        exp_approx = self.IntExpSTE.apply(x_shifted, self.M, self.N)
        
        sum_exp = exp_approx.sum(dim=-1, keepdim=True)
        
        ln_sum = self.IntLnSTE.apply(sum_exp, self.M+4, self.N)
        
        logits = x_shifted - ln_sum
        
        probs = self.IntExpSTE.apply(logits, self.M, self.N)
        return probs

    
"""
    Inerger only GeLU with LUT 
"""
class IntGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        indices = module._get_segment_indices(x)
        
        dtype = x.dtype
        slopes = module.slopes[indices].to(dtype)
        intercepts = module.intercepts[indices].to(dtype)
        low_threshold = module.LOW_THRESHOLD.to(dtype)
        up_threshold = module.UP_THRESHOLD.to(dtype)
        y = slopes * x + intercepts
        
        y = torch.where(x < module.LOW_THRESHOLD.to(x.dtype), torch.zeros_like(y), y)
        y = torch.where(x > module.UP_THRESHOLD.to(x.dtype), x, y) # GELU(x) -> x as x -> inf

        ctx.save_for_backward(slopes, x)
        ctx.low_threshold = module.LOW_THRESHOLD
        ctx.up_threshold = module.UP_THRESHOLD
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        slopes, x = ctx.saved_tensors
        
        grad_input = grad_output * slopes
        
        grad_input = torch.where(x < ctx.low_threshold, torch.zeros_like(grad_input), grad_input)
        grad_input = torch.where(x > ctx.up_threshold, torch.ones_like(grad_input) * grad_output, grad_input)
        
        return grad_input, None # module 不需要梯度

class IntGeLU_LUT(nn.Module):
    def __init__(self, M: int = 8, N: int = 8, p: int = 16, 
                 low_threshold: float = -4.0, up_threshold: float = 4.0):
        super().__init__()
        self.M, self.N = M, N
        self.p = p
        
        # Clamping thresholds
        self.register_buffer('LOW_THRESHOLD', torch.tensor(low_threshold))
        self.register_buffer('UP_THRESHOLD', torch.tensor(up_threshold))
        
        segments_raw = [
            (-0.0312, -0.0938, -4.0),   
            (-0.0938, -0.25,   -1.9844), 
            (0.0938,  -0.0938, -0.7617),  
            (0.375,   0.0,    -0.3359),  
            (0.625,   0.0,     -0.0039),    
            (0.875,  -0.0938,  0.375),  
            (1.0938,  -0.2500,     0.7539), 
            (1.0312,  -0.0938,     1.9766)     
        ]
        
        self.register_buffer('slopes', torch.tensor([s[0] for s in segments_raw]))
        self.register_buffer('intercepts', torch.tensor([s[1] for s in segments_raw]))
        self.register_buffer('thresholds', torch.tensor([s[2] for s in segments_raw]))

    def _get_segment_indices(self, x):
        indices = torch.bucketize(x, self.thresholds) - 1
        indices = torch.clamp(indices, 0, len(self.thresholds) - 1)
        return indices

    def forward(self, x):
        return IntGeLUFunction.apply(x, self)

""" 
    Int-LayerNorm Implementation from I-ViT 
"""
class IntLayerNorm(nn.LayerNorm):
    def __init__(self, 
                normalized_shape, 
                eps=1e-5,
                elementwise_affine=True):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer('norm_scaling_factor', torch.zeros(normalized_shape))  # 1
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x):  # , scaling_factor=None
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        scaling_factor = 1 / 2 ** 8
        x_int = x / scaling_factor
        mean_int = round_ste(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste((k + floor_ste(var_int/k))/2)
            k = k_1
        std_int = k

        factor = floor_ste((2 ** 31-1) / std_int)
        y_int = floor_ste(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30

        # scaling and shifting
        bias = self.bias / self.weight.data.detach()
        bias_int = floor_ste(bias / scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x  #, scaling_factor

class IntLayerNormOptimized(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, output_bit=8, quant_mode=True, force_dequant="none"):
        super().__init__()
        self.D = normalized_shape[0] if isinstance(normalized_shape, (list, tuple)) else normalized_shape
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(self.D))
        self.bias = nn.Parameter(torch.zeros(self.D))

        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "layernorm"]:
            self.quant_mode = False

        self.register_buffer("shift", torch.tensor([0], dtype=torch.long))
        self.output_bit = output_bit
        self.max_bit = 32 # Hardware accumulator limit
        
        # --- Hardware Approximation Configurations ---
        self.Q_D = 16
        self.inv_D = int(round((1 << self.Q_D) / self.D))
        
        self.m = 4        # Number of mantissa bits for LUT index
        self.Q_lut = 15   # Fractional bits for LUT values
        self._init_isqrt_lut()

        self.Q_w = 8      # Affine transform precision
        
    def _init_isqrt_lut(self):
        """
        Pre-computes the Inverse Square Root LUT.
        Size = 2^(m+1). Index is formed by: [parity of k (1 bit)] | [mantissa (m bits)]
        """
        lut_size = 1 << (self.m + 1)
        lut = torch.zeros(lut_size, dtype=torch.long)
        for i in range(lut_size):
            parity = i >> self.m
            mantissa = i & ((1 << self.m) - 1)
            # Reconstruct the normalized value: 2^parity * (1 + mantissa / 2^m)
            val = (2 ** parity) * (1.0 + mantissa / (1 << self.m))
            # Store scaled ISqrt value
            lut[i] = int(round((1 << self.Q_lut) / math.sqrt(val)))
        self.register_buffer("isqrt_lut", lut)

    def set_shift(self, y_int):
        """ Adjusts dynamic shift using L_infinity norm to mathematically guarantee no overflow """
        with torch.no_grad():
            # Target: D * (y_max >> S)^2 < 2^max_bit
            # => y_max >> S < sqrt(2^max_bit / D)
            y_max = y_int.abs().max().float()
            max_val_allowed = math.sqrt(((1 << self.max_bit) - 1) / self.D)
            
            if y_max > max_val_allowed:
                # Calculate required shift to keep max squared sum strictly within max_bit
                required_shift = torch.ceil(torch.log2(y_max / max_val_allowed)).long()
                self.shift = torch.max(self.shift, required_shift)

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            mean = x.mean(dim=2, keepdim=True)
            y = x - mean
            var = torch.mean(y**2, dim=2, keepdim=True)
            return (y / torch.sqrt(self.eps + var)) * self.weight + self.bias, None

        # --- Hardware-Equivalent Integer Forward Pass ---
        if scaling_factor is not None:
            x_int = (x / scaling_factor).round().long() # Adding round() reduces quantization noise
        else:
            x_int = x.round().long()

        # [Step 1] Division-free Mean Calculation
        sum_x = x_int.sum(dim=2, keepdim=True)
        mean_int = torch.bitwise_right_shift(sum_x * self.inv_D, self.Q_D)
        
        y_int = x_int - mean_int

        # [Step 2] Variance Calculation with Dynamic Overflow Prevention
        y_shifted = torch.bitwise_right_shift(y_int, self.shift.item())
        y_sq_int = y_shifted * y_shifted
        sum_sq = y_sq_int.sum(dim=2, keepdim=True)
        
        # Check accumulator overflow on SUM OF SQUARES, not variance!
        if self.training and sum_sq.max() >= (1 << self.max_bit):
            self.set_shift(y_int)
            y_shifted = torch.bitwise_right_shift(y_int, self.shift.item())
            sum_sq = (y_shifted * y_shifted).sum(dim=2, keepdim=True)
            
        var_int = torch.bitwise_right_shift(sum_sq * self.inv_D, self.Q_D)
        var_int = torch.clamp(var_int, min=1) # Prevents log2(0)

        # [Step 3] LOD & LUT-based Inverse Square Root
        k = var_int.float().log2().long() 
        parity = k & 1
        mask = (1 << self.m) - 1
        
        shift_amt = k - self.m
        mantissa = torch.where(
            shift_amt >= 0,
            torch.bitwise_right_shift(var_int, shift_amt) & mask,
            torch.bitwise_left_shift(var_int, -shift_amt) & mask
        )
        
        idx = (parity << self.m) | mantissa
        inv_sqrt_val = self.isqrt_lut[idx]

        p = torch.bitwise_right_shift(k, 1)
        
        total_shift = p + self.shift.item() + self.Q_lut
        
        # Quantize Weights and Bias 
        weight_int = (self.weight * (1 << self.Q_w)).round().long()
        bias_int = (self.bias * (1 << self.Q_w)).round().long() 
        
        # Fusion: Y = Y_int * ISqrt * W
        scaled_y = y_int * inv_sqrt_val * weight_int 
        
        out_int = torch.bitwise_right_shift(scaled_y, total_shift) + bias_int
        
        out_fp32 = out_int.float() / (1 << self.Q_w)

        new_scaling_factor = scaling_factor if scaling_factor is not None else 1.0
        
        return out_fp32#, new_scaling_factor
