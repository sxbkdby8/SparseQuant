import torch
import torch.nn as nn
from typing import Dict
import types

def replace_linear_with_(model, new_class, exclude=[], **kwargs):
    """replace linear layers with new_class in a model. It's a inplace operation."""
    for name, module in model.named_children():
        if module in exclude:
            continue
        if isinstance(module, torch.nn.Linear):
            new_linear = new_class(module.in_features, module.out_features, module.bias is not None, **kwargs)
            new_linear.weight.data = module.weight.data
            if module.bias is not None:
                new_linear.bias.data = module.bias.data
            new_linear.to(module.weight.device)
            setattr(model, name, new_linear)   
        else:
            replace_linear_with_(module, new_class, exclude)     
    return model

def dyna_set_sparse_budget(model: nn.Module, budget: Dict[str, int], sparse_m:int=4, verbose: bool = False):
    from .sparse import SparseLinear
    for name, m in model.named_modules():
        if isinstance(m, SparseLinear):
            for target_name, n_value in budget.items():
                if target_name in name:
                    # n_value = 4 - n_value
                    # n_value = 3
                    m._reset_n_m_params(n_value, sparse_m)
                    if verbose:
                        print(f"[SparseConfig] {name} set to {n_value}:{sparse_m} sparsity")  
                        

class ActivationCaptureHook:
    """
    Hook to capture the input activations and original floating-point 
    outputs of specific non-linear modules.
    """
    def __init__(self):
        self.inputs = []
        self.outputs = []
        
    def __call__(self, module, input, output):
        # Detach and move to CPU to save GPU memory during capture
        self.inputs.append(input[0].detach())
        self.outputs.append(output.detach())

def register_eval_hooks(model):
    """
    Register hooks on GELU, Softmax, and LayerNorm modules.
    Returns a dictionary containing the hooks and module info.
    """
    hook_dict = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.GELU, nn.LayerNorm)):
            hook = ActivationCaptureHook()
            handle = module.register_forward_hook(hook)
            
            # Store module type and the original module instance (needed for LayerNorm weights)
            hook_dict[name] = {
                'hook': hook,
                'type': type(module),
                'orig_module': module
            }
            handles.append(handle)
        elif type(module).__name__ == 'Attention':
            softmax_name = f"{name}.softmax"
            hook_dict[softmax_name] = {
                'type': 'Softmax',
                'inputs': [],
                'outputs': []
            }
            
            def make_custom_forward(mod, s_name):
                def custom_forward(self, x, *args, **kwargs):  # attn_mask ignored in following forward func
                    B, N, C = x.shape
                    
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)

                    attn_logits = (q @ k.transpose(-2, -1)) * self.scale
                    
                    hook_dict[s_name]['inputs'].append(attn_logits.detach().cpu())
                    
                    attn_probs = attn_logits.softmax(dim=-1)
                    
                    # ---> CAPTURE POST-SOFTMAX HERE <---
                    hook_dict[s_name]['outputs'].append(attn_probs.detach().cpu())
                    
                    attn = self.attn_drop(attn_probs)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                    return x
                
                return custom_forward

            module.forward = types.MethodType(make_custom_forward(module, softmax_name), module)
    return hook_dict, handles

