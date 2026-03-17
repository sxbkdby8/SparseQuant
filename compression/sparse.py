import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import itertools
from .quant import Round, FunLSQ

def generate_N_M_masks(N, M):
    # Create all possible binary combinations N:M sparse masks
    combinations = list(itertools.combinations(range(M), N))
    # Create a tensor to store the result
    result = torch.zeros((len(combinations), M), dtype=torch.float32)
    # Fill in the ones according to the combinations
    for i, indices in enumerate(combinations):
        result[i, torch.tensor(indices)] = 1
    return result

class SparseLinearFrozen(nn.Linear):
    """A linear layer with a fixed mask that is not updated during training."""
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinearFrozen, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None})"

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class SparseLinear(nn.Linear):
    """A linear layer with a learnable mask that sparsifies the weights."""
    def __init__(self, in_features, out_features, bias=True, N=2, M=4, gate_init_std=0.2, tau=1, hard=False, scaling=1):
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.tau = 1
        self.scaling = scaling
        self.hard = hard
        self.register_buffer('mask', torch.ones((out_features, in_features), dtype=torch.float32))
        self._reset_n_m_params(N, M, gate_init_std)
        # self.N = N
        # self.M = M 
        # self._mask_options = generate_N_M_masks(N, M) 
        # self.gate = nn.Parameter(torch.empty(
        #         self.weight.numel()//M, self._mask_options.size(0), device=self.weight.device, dtype=self.weight.dtype), requires_grad=True)
        # torch.nn.init.normal_(self.gate, mean=0, std=gate_init_std)
        # self.mask_oudated = False
    
    def _reset_n_m_params(self, N, M=4, gate_init_std=0.2):
        self.N, self.M = N, M
        self._mask_options = generate_N_M_masks(N, M)
        self.gate = nn.Parameter(torch.empty(
            self.weight.numel()//M, self._mask_options.size(0), device=self.weight.device, dtype=self.weight.dtype), requires_grad=True)
        torch.nn.init.normal_(self.gate, mean=0, std=gate_init_std)
        self.mask_oudated = False
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, bias={self.bias is not None}, N={self.N}, M={self.M}, tau={self.tau}, scaling={self.scaling}, hard={self.hard})"

    def sparse_weight_reg(self):
        return self._sparse_weight_reg

    def _update_hard_mask(self):
        return self._mask_options[torch.argmax(self.gate, dim=1)].view(self.out_features, self.in_features)
    
    def forward(self, x):
        if self.training:
            self.mask_oudated = True # reset selected since we will update it
            soft_index = F.gumbel_softmax(self.gate * self.scaling, tau=self.tau, hard=self.hard, dim=1) # (Blocks x Candidate Masks)
            soft_mask = soft_index @ self._mask_options.to(x.device) # (Blocks x Candidate Masks) @ (Candidate Masks x M) = (Blocks x M)
            soft_mask = soft_mask.view(self.out_features, self.in_features)
            self._sparse_weight_reg = (self.weight.detach() * soft_mask).pow(2).sum()
            return F.linear(x, soft_mask * self.weight, self.bias)
        else:
            if self.mask_oudated: # for inference, we only compute the winner masks once for efficiency
                self._mask_options = self._mask_options.to(x.device)
                # self.mask = self._mask_options[torch.argmax(self.gate, dim=1)].view(self.out_features, self.in_features)
                self.mask = self._update_hard_mask()
                self.mask_oudated = False
            return F.linear(x, self.mask * self.weight, self.bias)

    def load_mask_prior(self, prior_strength=3):
        with torch.no_grad():
            sparsity = (self.mask==0).sum().item() / self.mask.numel()
            # prior will be the inner product the different candidates to the prior mask
            priors = (self._mask_options.unsqueeze(0) * self.mask.view(-1, 1, 4)).sum(dim=2) # (1, Candidate Masks, M) * (Blocks, 1, M) => Blocks x Candidate Masks
            self.gate.data += (priors-self.N//2) * self.gate.std() * prior_strength

            if torch.distributed.get_rank() == 0:
                print(f"initializing with prior (strength={prior_strength}), Prior Sparsity: {sparsity}")
                print(f"mean: {self.gate.mean().item()}, std: {self.gate.std().item()}, max: {self.gate.max().item()}, min: {self.gate.min().item()}")
                

class SparseQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bits=4, per_channel=False, sym=True):
        super().__init__(in_features, out_features, bias)
        
        if sym:
            self.Qn = -(2**(bits - 1))
            self.Qp = 2**(bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**bits - 1
        self.per_channel = per_channel
        
        # LSQ Step-size parameters (alphas)
        # self.weight_alpha = nn.Parameter(torch.ones((out_features, 1) if per_channel else (1,)))
        self.weight_alpha = torch.nn.Parameter(torch.ones((self.weight.size()[0],) if self.per_channel else 1.0), requires_grad=True)
        self.act_alpha = nn.Parameter(torch.tensor(1.0))
        
        self.register_buffer('init_state', torch.zeros(1)) 
        # self.mask = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('mask', torch.empty(out_features, in_features))
    
    def forward(self, x):
        if self.training and self.init_state == 0:
            self.act_alpha.data.copy_(2 * x.detach().abs().mean() / math.sqrt(self.Qp))
            if self.per_channel:
                # w_mean = self.weight.detach().abs().view(self.out_features, -1).mean(dim=1, keepdim=True)
                weight_tmp = self.weight.detach().contiguous().view(self.weight.size()[0], -1)
                self.weight_alpha.data.copy_(torch.mean(torch.abs(weight_tmp), dim=1)*2/(math.sqrt(self.Qp)))
            else:
                self.weight_alpha.data.copy_(2 * self.weight.detach().abs().mean() / math.sqrt(self.Qp))
            self.init_state.fill_(1)
        if not hasattr(self, 'g_w') or not hasattr(self, 'g_a'):
            self.g_w = 1.0 / math.sqrt(self.weight.numel() * self.Qp)
            self.g_a = 1.0 / math.sqrt(x.numel() * self.Qp)

        x_q = FunLSQ.apply(x, self.act_alpha, self.g_a, self.Qn, self.Qp, False)

        # 4. Apply Mask and Quantize Weights
        masked_weight = self.weight * self.mask
        w_q = FunLSQ.apply(masked_weight, self.weight_alpha, self.g_w, self.Qn, self.Qp, self.per_channel)

        # 5. Standard Linear Operation
        return F.linear(x_q, w_q, self.bias)
        
       

"""
    MatMul Impl
"""
class QuantMatmul(nn.Module):
    def __init__(self, n_bits: int, batch_init=8):
        super().__init__()
        self.n_bits = n_bits
        self.batch_init = batch_init
        
        if n_bits > 0:
            self.a_alpha = nn.Parameter(torch.ones(1))
            self.b_alpha = nn.Parameter(torch.ones(1))
            
            self.Qn = - 2 ** (n_bits - 1)
            self.Qp = 2 ** (n_bits - 1) - 1
                
            self.register_buffer('init_state', torch.zeros(1))

    def _get_g(self, x):
        return 1.0 / math.sqrt(x.numel() * self.Qp)

    def _initialize_alpha(self, x, alpha_param):
        with torch.no_grad():
            std = x.std()
            mean = x.abs().mean()
            initial_alpha = 2 * mean / math.sqrt(self.Qp)
            alpha_param.data.fill_(initial_alpha)

    def forward(self, A, B, transpose_b: bool = False):
        if self.n_bits <= 0:
            if transpose_b:
                return torch.matmul(A, B.transpose(-2, -1))
            return torch.matmul(A, B)

        if self.training and self.init_state == 0:
            self._initialize_alpha(A, self.a_alpha)
            self._initialize_alpha(B, self.b_alpha)
            self.init_state += 1

        g_a = self._get_g(A)
        g_b = self._get_g(B)
            
        A_q = FunLSQ.apply(A, self.a_alpha, g_a, self.Qn, self.Qp)
        B_q = FunLSQ.apply(B, self.b_alpha, g_b, self.Qn, self.Qp)

        if transpose_b:
            return A_q @ B_q.transpose(-2, -1)
        else:
            return A_q @ B_q
        # if transpose_b:
        #     return torch.matmul(A_q, B_q.transpose(-2, -1))
        # else:
        #     return torch.matmul(A_q, B_q)

