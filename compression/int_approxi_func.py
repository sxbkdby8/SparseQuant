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
        # bias = self.bias.data.detach() / (self.weight.data.detach())
        bias = self.bias / self.weight.data.detach()
        bias_int = floor_ste(bias / scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x  #, scaling_factor
