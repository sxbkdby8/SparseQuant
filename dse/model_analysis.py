## Sparsed Model Analysis ##
## Including FLOPS, Params etc. ## 

import torch 
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

from compression import SparseLinear, SparseLinearFrozen, SparseQuantLinear, QuantMatmul

class DeiTConfig:
    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        patch_size=16,
        image_size=224,
        mlp_ratio=4,
        num_classes=1000,
        prune_layers=[3, 6, 9],
        prune_tokens=[0, 0, 0], 
        num_prefix_tokens=1
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.prune_layers = prune_layers    
        self.prune_tokens = prune_tokens 
        
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_length = self.num_patches + num_prefix_tokens
        
        self.layer_seq_lengths = self._compute_layer_seq_lengths()
        
    def _compute_layer_seq_lengths(self):
        layer_seq_lengths = [self.seq_length]  
        for i in range(1, self.num_layers + 1):
            prev_seq_length = layer_seq_lengths[-1]
            if i in self.prune_layers:  
                prune_index = self.prune_layers.index(i)
                tokens_to_prune = self.prune_tokens[prune_index]
                new_seq_length = prev_seq_length - tokens_to_prune
                new_seq_length = max(1, new_seq_length)
            else:
                new_seq_length = prev_seq_length
            layer_seq_lengths.append(new_seq_length)
        return layer_seq_lengths


def compute_conv2d_metrics(module:nn.Conv2d, input_shape: Tuple):
    assert len(input_shape) == 4
    batch_size, in_channels, height, width = input_shape
    out_channels = module.out_channels
    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding
    groups = module.groups
    
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h = kernel_w = kernel_size
    
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride
    
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding
    
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    output_spatial = out_h * out_w
    
    weight_params = (in_channels // groups) * out_channels * kernel_h * kernel_w
    non_zero_weight_params = weight_params  # For dense conv layers
    
    # Add bias if exists
    bias_params = out_channels if module.bias is not None else 0
    total_params = weight_params + bias_params
    non_zero_params = non_zero_weight_params + bias_params
    
    # Calculate FLOPs (considering batch size and output spatial dimensions)
    dense_flops = 2 * kernel_h * kernel_w * (in_channels // groups) * out_channels * output_spatial * batch_size
    
    if module.bias is not None:
        dense_flops += out_channels * output_spatial * batch_size
    
    # For dense convolution, sparse FLOPs = dense FLOPs
    sparse_flops = dense_flops
    
    return total_params, non_zero_params, dense_flops, sparse_flops

def compute_matmul_metrics(matmul_layer:QuantMatmul, 
                            input_shape: Optional[Tuple] = None):
    assert isinstance(matmul_layer, QuantMatmul), \
        f"Layer should be QuantMatmul, got {type(matmul_layer)}"
    assert input_shape is not None , \
        "Input shapes must be provided for QuantMatmul"
    assert len(input_shape) == 3 , f"Input shape should be (batch, m, n)"
    # QuantMatmul has no trainable parameters except quantization parameters
    param_count = 2 if getattr(matmul_layer, 'n_bits', 0) > 0 else 0
    non_zero_params = param_count  # All parameters are non-zero
    
    batch, m, n = input_shape
    if getattr(matmul_layer, 'transpose_a', False):   # 简单处理喽
        p = m 
    else:
        p = n    
    
    dense_flops = 2 * m * n * p * batch
    n_bits = getattr(matmul_layer, 'n_bits', 32)
    sparse_flops = dense_flops
    
    return param_count, non_zero_params, dense_flops, sparse_flops
    
def count_sparse_linear_metrics(module: Union[nn.Linear, SparseLinear, SparseLinearFrozen, SparseQuantLinear],
                                input_shape: Tuple,
                                sparsity_ratio=1.0) -> Tuple[int, int, int, int]:
    """Count metrics for sparse linear layers"""
    assert len(input_shape) == 3, f"Input shape: (batch_size, seq_length, hidden_size) "
    batch_size, seq_len = input_shape[:2]
    in_features = module.in_features
    out_features = module.out_features
    weight_params = in_features * out_features
    
    if hasattr(module, 'mask') and module.mask is not None:
        # For frozen sparse layers
        mask = module.mask
        # non_zero_weight_params = int((mask != 0).sum().item())
        non_zero_weight_params = int(weight_params * sparsity_ratio)
        # sparsity_ratio = 1.0 - (non_zero_weight_params / weight_params)
    else:
        non_zero_weight_params = weight_params
        # sparsity_ratio = 0.0
    
    bias_params = out_features if module.bias is not None else 0
    total_params = weight_params + bias_params
    non_zero_params = non_zero_weight_params + bias_params
    
    # FLOPs calculation with token pruning
    num_tokens = batch_size * seq_len
    dense_flops = 2 * in_features * out_features * num_tokens
    sparse_flops = 2 * non_zero_weight_params * num_tokens
    
    if module.bias is not None:
        dense_flops += out_features * num_tokens
        sparse_flops += out_features * num_tokens
    
    return total_params, non_zero_params, dense_flops, sparse_flops


def analysis_sparse_model(model: nn.Module, 
                          config, 
                         sparse_budget:Dict[str, int]=None) -> Dict[str, Any]:
    import re
    _PROC_MAPPING  = {
        nn.Conv2d : compute_conv2d_metrics,
        QuantMatmul : compute_matmul_metrics,
        SparseLinear : count_sparse_linear_metrics,
        SparseLinearFrozen : count_sparse_linear_metrics,
        SparseQuantLinear : count_sparse_linear_metrics,
        nn.Linear : count_sparse_linear_metrics,
    }
    ## init Global Variable
    total_model_params = 0
    total_non_zero_params = 0
    total_dense_flops = 0
    total_sparse_flops = 0
    layer_metrics = {}
    
    batch_size = 1
    
    for name, m in model.named_modules():
        if not isinstance(m, tuple(_PROC_MAPPING.keys())):
            continue

        if isinstance(m, nn.Conv2d):
            seq_len = 0
            input_shape = (batch_size, 3, config.image_size, config.image_size)
        elif 'qkv' in name.lower() or 'attn.proj' in name.lower() or isinstance(m, QuantMatmul):
            match = re.search(r'blocks\.(\d+)', name)
            transformer_layer_idx = int(match.group(1))
            seq_len = config.layer_seq_lengths[transformer_layer_idx]
            input_shape = (batch_size, seq_len, config.hidden_size)
        elif 'mlp' in name.lower():
            match = re.search(r'blocks\.(\d+)', name)
            transformer_layer_idx = int(match.group(1))
            pseudo_idx= max(transformer_layer_idx+1, config.num_layers-1)
            seq_len = config.layer_seq_lengths[pseudo_idx]
            if 'fc1' in name.lower():
                input_shape = (batch_size, seq_len, config.hidden_size)
            elif 'fc2' in name.lower():
                input_shape = (batch_size, seq_len, config.hidden_size * config.mlp_ratio)
        elif 'head' in name and isinstance(m, nn.Linear):
            input_shape = (batch_size, 1, config.hidden_size)
        else:
            # input_shape = (batch_size, config.hidden_size)
            raise NotImplementedError(f"Layer: {name} not implemented")
        
        # layer_metrics[name] = _PROC_MAPPING[type(m)](m, input_shape)
        sp_mapping = {1: 0.75, 2:0.5, 3:0.25}
        args = (m, input_shape)
        if sparse_budget is not None and name in sparse_budget.keys():
            args = (m, input_shape, sp_mapping[sparse_budget[name]])
            
        total_params, non_zero_params, dense_flops, sparse_flops = _PROC_MAPPING[type(m)](*args)
        layer_metrics[name] = {
                        'name': f"{name}",
                        'params': total_params,
                        'non_zero_params': non_zero_params,
                        'dense_flops': dense_flops,
                        'sparse_flops': sparse_flops,
                        'seq_length': seq_len,
                        'input_shape': input_shape
                    }
    # Gather Results
    for lm in layer_metrics.values():
        total_model_params += lm['params']
        total_non_zero_params += lm['non_zero_params']
        total_dense_flops += lm['dense_flops']
        total_sparse_flops += lm['sparse_flops']
    
    return {
        'total_model_params': total_model_params,
        'total_non_zero_params': total_non_zero_params,
        'total_dense_flops': total_dense_flops,
        'total_sparse_flops': total_sparse_flops,
        'layer_metrics': layer_metrics
    }
    
        # if isinstance(m, _PROC_MAPPING.keys()):
        #     total_params, non_zero_params, dense_flops, sparse_flops = _PROC_MAPPING[type(m)](m, input_shapes[name], batch_size)
