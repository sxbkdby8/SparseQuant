import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class TokenSelector(nn.Module):
    def __init__(self, sel_method: str = 'topk', reduced_token: int = 32, prefix_token: int = 1):
        super().__init__()
        self.sel_method = sel_method
        self.reduce_token = reduced_token
        self.prefix_token = prefix_token
        
        # Validate parameters
        if sel_method not in ['topk']:
            raise ValueError(f"Unsupported selection method: {sel_method}")
        if reduced_token < 0:
            raise ValueError(f"reduced_token must be non-negative, got {reduced_token}")
        if prefix_token < 0:
            raise ValueError(f"prefix_token must be non-negative, got {prefix_token}")
            
    def select_topk_tokens(self, x: torch.Tensor, attn: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if self.reduce_token == 0:
            return x, None, None, -1  # No tokens to reduce
            
        # Validate input shapes
        if len(x.shape) != 3:
            raise ValueError(f"Expected x with shape (B, L, C), but got {x.shape}")
            
        if attn is None:
            raise ValueError("Attention tensor is required for topk selection method")
            
        if len(attn.shape) != 4:
            raise ValueError(f"Expected attn with shape (B, H, L, L), but got {attn.shape}")
            
        B, L, C = x.shape
        H = attn.shape[1]
        
        # Calculate how many tokens to keep after reduction
        tokens_to_keep = L - self.reduce_token
        if tokens_to_keep <= self.prefix_token:
            raise ValueError(
                f"Insufficient tokens after reduction: "
                f"seq_len={L}, reduced_token={self.reduce_token}, "
                f"tokens_to_keep={tokens_to_keep}, prefix_token={self.prefix_token}. "
                f"Must keep at least {self.prefix_token + 1} tokens."
            )
            
        # Calculate how many non-prefix tokens to keep
        non_prefix_to_keep = tokens_to_keep - self.prefix_token
        
        cls_attn = attn[:, :, 0, self.prefix_token:]
        cls_attn = cls_attn.mean(dim=1)  # Shape: (B, L - prefix_token)
        
        _, idx_rel = torch.topk(
            cls_attn, 
            k=non_prefix_to_keep, 
            dim=1, 
            largest=True, 
            sorted=False  # Changed to False for efficiency unless ordering is needed
        )
        

        idx_abs = idx_rel + self.prefix_token
        
        prefix_indices = torch.arange(self.prefix_token, device=x.device)
        prefix_indices = prefix_indices.unsqueeze(0).expand(B, -1)  # Shape: (B, prefix_token)
        full_indices = torch.cat([prefix_indices, idx_abs], dim=1)
        full_indices, _ = torch.sort(full_indices, dim=1)
        
        return x, full_indices, cls_attn, tokens_to_keep
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if self.sel_method == 'topk':
            attn = kwargs.get('attn')
            return self.select_topk_tokens(x, attn)
        else:
            # For other methods (if added in future), return x with placeholders
            return x, None, None, -1


class TokenPruner(nn.Module):
    def __init__(self, selector: TokenSelector):
        super().__init__()
        self.selector = selector
        
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        x_full, indices, cls_attn, tokens_to_keep = self.selector(x, **kwargs)
        
        if indices is None:
            return x_full, {}
        
        B, L, C = x_full.shape
        batch_indices = torch.arange(B, device=x.device).view(B, 1).expand(-1, tokens_to_keep)
        x_selected = x_full[batch_indices, indices, :]
        metadata = {
            'selected_indices': indices,
            'cls_attention': cls_attn,
            'original_length': L,
            'pruned_length': tokens_to_keep,
            'reduction_ratio': 1.0 - tokens_to_keep / L
        }
        
        return x_selected, metadata