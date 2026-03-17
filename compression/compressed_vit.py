import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function 

import timm
from timm.layers import (
    Attention, 
    Mlp,
    trunc_normal_,
    LayerNorm,
    DropPath,
    PatchEmbed,)
from timm.models.vision_transformer import Block

from typing import Optional, Type, Union, Tuple, Dict, List
from functools import partial

from .sparse import SparseLinear, SparseQuantLinear, QuantMatmul
from .int_approxi_func import IntSoftmaxFixed, IntLayerNorm, IntGeLU_LUT
from .token_pruner import TokenPruner, TokenSelector

class CompressedAttention(Attention):
    def __init__(
        self, 
        dim: int,
        softmax_module: nn.Module,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Optional[Type[nn.Module]] = None,
        n_bits: int = 8,
    ):
        super().__init__(dim, num_heads, qkv_bias, qk_norm, 
                        attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.qkv = SparseQuantLinear(dim, dim * 3, bias=qkv_bias, bits=n_bits, per_channel=True)
        self.proj = SparseQuantLinear(dim, dim, bias=proj_bias, bits=n_bits, per_channel=True)
        self.softmax = softmax_module   
        self.matmul_qk = QuantMatmul(n_bits=n_bits)
        self.matmul_v = QuantMatmul(n_bits=n_bits)
    
    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Replace torch.matmul with QuantMatmul
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        attn = self.matmul_qk(q* self.scale, k, transpose_b=True) 
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Replace torch.matmul with QuantMatmul
        x = self.matmul_v(attn, v)
        # x = attn @ v           
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn:
            return x, attn
        return x
    
class CompressedMlp(Mlp):
    def __init__(
        self, 
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        n_bits: int = 8,
        drop: float = 0.,
        bias: bool = True,
    ):
        super().__init__(in_features, hidden_features, out_features, act_layer, drop=drop, bias=bias)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = SparseQuantLinear(in_features, hidden_features, bias=bias, bits=n_bits, per_channel=True)
        self.fc2 = SparseQuantLinear(hidden_features, out_features, bias=bias, bits=n_bits, per_channel=True)   


class CompressedBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_bits: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        softmax_layer: Type[nn.Module] = nn.Softmax,
        sel_method: str = 'topk',
        reduced_token: int = 0, # Number of tokens to prune in this block
        prefix_token: int = 1,
        **kwargs
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, 
                         proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, 
                         act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = CompressedAttention(
            dim, softmax_module=softmax_layer(),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop, n_bits=n_bits
        )
        
        self.norm2 = norm_layer(dim)
        self.mlp = CompressedMlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, n_bits=n_bits, drop=proj_drop, bias=proj_bias
        )

        # Token Pruning Setup
        self.pruner = None
        if reduced_token > 0:
            selector = TokenSelector(sel_method, reduced_token, prefix_token)
            self.pruner = TokenPruner(selector)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Residual
        res, attn_weights = self.attn(self.norm1(x), return_attn=True)
        x = x + self.drop_path1(res)
        
        # Token Pruning occurs after Attention because it needs attention weights
        if self.pruner is not None:
            x, _ = self.pruner(x, attn=attn_weights)
            
        # MLP + Residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    
def init_compressedvit_from_sparsevit(target_model, ori_state_dict):
    target_state_dict = target_model.state_dict()
    
    sparse_quant_modules = {}

    for name, m in target_model.named_modules():
        if isinstance(m, SparseQuantLinear):
            sparse_quant_modules[name] = m
    
    for name, m in target_model.named_modules():
        # if name in ori_state_dict.keys():
        if isinstance(m, SparseQuantLinear):  # Fix the mask and copy other parameters
            prefix = name + '.' #  name.split('.')[0]
            weight_key = prefix + 'weight'
            if weight_key in ori_state_dict:
                ori_weight = ori_state_dict[weight_key]
                
                mask_key = prefix + 'mask'
                if mask_key in ori_state_dict:
                    hard_mask = ori_state_dict[mask_key]
                else:
                    print(f"[SparseConfig] Warning: Layer: {name} has no mask, try to generate by calling _update_hard_mask instead")
                    # mask
                    try:
                        hard_mask = m._update_hard_mask()
                    except:
                        print(f"[SparseConfig] Error: Failed to update hard mask for layer: {name}")

                m.weight.data.copy_(ori_weight)
                m.mask.data.copy_(hard_mask)
            else:
                print(f"[SparseConfig] Warning: Weight key {weight_key} not found in original state_dict")
                
            bias_key = prefix + 'bias'
            if bias_key in ori_state_dict and m.bias is not None:
                m.bias.data.copy_(ori_state_dict[bias_key])
            
        # Other Case, including Linear of LayerNorm Patch Embedding etc.
        elif hasattr(m, 'weight') or hasattr(m, 'bias'):
            for param_name, param in m.named_parameters(recurse=False):
                full_key = name + '.' + param_name if name else param_name
                
                if full_key in ori_state_dict:
                    try:
                        param.data.copy_(ori_state_dict[full_key])
                    except Exception as e:
                        print(f"[SparseConfig] Warning: Failed to parameter {param_name}:{full_key}, {e}")
                else:
                    raise ValueError(f"[SparseConfig] Error: Parameter {param_name}:{full_key} not found in original state_dict")
        
    print("[SparseConfig] Model initialization complete")
    return target_model
                    
def create_compressed_deit_small(
    pretrained: bool = False, 
    n_bits: int = 8, 
    reduced_token: int = 40, 
    pruning_layer_indices = [3, 6, 9],
    pt_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    **kwargs
):
    from timm.models.deit import VisionTransformerDistilled
    
    # softmax_layer = partial(IntSoftmaxFixed, 8, 8, 20)
    softmax_layer = partial(nn.Softmax, dim=-1)
    norm_layer = IntLayerNorm 
    # norm_layer = nn.LayerNorm
    act_layer = IntGeLU_LUT
    # act_layer = nn.GELU
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
                      norm_layer=norm_layer, act_layer=act_layer)
    model_args.update(kwargs)
    
    # State tracking for the block factory
    current_layer_idx = 0
    def compressed_block_fn(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, 
                          proj_drop, attn_drop, drop_path, act_layer, norm_layer, **block_kwargs):
        nonlocal current_layer_idx
        
        # Determine if the current layer requires a TokenPruner
        # reduced_token = reduced_token if current_layer_idx in pruning_layer_indices else 0
        tp = reduced_token if current_layer_idx in pruning_layer_indices else 0
        
        # Instantiate CompressedBlock with quantized sub-modules
        block = CompressedBlock(
            dim=dim,
            num_heads=num_heads,
            n_bits=n_bits,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=IntGeLU_LUT,      
            norm_layer=IntLayerNorm, 
            softmax_layer=softmax_layer,
            reduced_token=tp,
            sel_method='topk',           
            prefix_token=2,              
            **block_kwargs
        )
        
        current_layer_idx += 1
        return block
    
    model = VisionTransformerDistilled(
        **model_args,
        block_fn=compressed_block_fn,
        **kwargs
    )
    
    
    if pt_state_dict is not None:
        init_compressedvit_from_sparsevit(model, pt_state_dict)
    elif pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['deit_small_distilled_patch16_224']))
        ## FIXME : Load state dict
        raise NotImplementedError
    
    return model

