import time 
import torch 
import torch.nn as nn 
import argparse
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import os
from typing import Any, Dict, List, Optional, Union
import pickle

import timm
import compression
from compression.sparsegpt import SparseGPT
from compression import dyna_set_sparse_budget

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def prune_magnitude(model, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=0.0, reverse=False):
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear)) and hasattr(layer, 'mask'):
            W = layer.weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
            if reverse:
                layer.mask.data = W_mask
            else:
                layer.mask.data = ~W_mask
            print(f"Layer {name} Sparsity: {torch.sum(W_mask).item()/W.numel()}")

@torch.no_grad()
def prune_magnitude_with_budget(
    model, 
    sparse_budget: dict = None, 
    prune_m: int = 4, 
    reverse: bool = False
):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and hasattr(layer, 'mask'):
            W = layer.weight.data
            W_metric = torch.abs(W)
            device = W.device
            
            if sparse_budget is not None and name in sparse_budget:
                prune_n = sparse_budget[name]
                
                if W_metric.shape[1] % prune_m != 0:
                    print(f"Warning: Layer {name} input features {W_metric.shape[1]} is not divisible by {prune_m}. Skipping N:M.")
                    continue
                
                W_reshaped = W_metric.view(W_metric.shape[0], -1, prune_m)
                _, indices = torch.topk(W_reshaped, k=prune_n, dim=-1, largest=False)
                W_mask = torch.zeros_like(W_reshaped, dtype=torch.bool)
                W_mask.scatter_(dim=-1, index=indices, value=True)
                W_mask = W_mask.view(W_metric.shape)
                
            else:
                print(f"Layer {name} does not have sparse budget. Skipping magnitude pruning.")
                continue

            if reverse:
                layer.mask.data.copy_(W_mask)
            else:
                layer.mask.data.copy_(~W_mask)
            
            actual_sparsity = torch.sum(W_mask).item() / W.numel()
            print(f"Layer: {name:40s} | Prune N:M = {sparse_budget.get(name, 0)}:{prune_m} | Actual Sparsity: {actual_sparsity:.4f}")

@torch.no_grad()
def prune_wanda(model, loader, nsamples=128, batch_size=1, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=0.0):
    class InputOutputHook():
        def __init__(self, layer):
            self.layer = layer
            self.columns = layer.weight.data.shape[1]
            self.dev = self.layer.weight.device

            self.hook = layer.register_forward_hook(self.hook_fn)
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
            self.nsamples = 0
        
        @torch.no_grad()
        def hook_fn(self, module, input, output):
            inp = input[0]
            if isinstance(self.layer, nn.Linear):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            self.scaler_row *= self.nsamples / (self.nsamples + inp.shape[0])
            self.nsamples += inp.shape[0]
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
        def close(self):
            self.hook.remove()

    nbatches = nsamples // batch_size

    layer_hook_pairs = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear)) and hasattr(layer, 'mask'):
            hook = InputOutputHook(layer)
            layer_hook_pairs.append((name, layer, hook))

    for i, (x, y) in enumerate(loader):
        if i == nbatches: break
        print(f"Batch {i+1}/{nbatches}")
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            _ = model(x)

    hook.close()
    for name, layer, hook in layer_hook_pairs:
        W_metric = torch.abs(layer.weight.data) * torch.sqrt(hook.scaler_row.reshape((1,-1)))
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        # structured n:m sparsity
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        layer.mask.data = ~W_mask
        print(f"Layer {name} Sparsity: {torch.sum(W_mask).item()/W_mask.numel()}")
        
@torch.no_grad()
def prune_sparsegpt(model, loader, nsamples=128, batch_size=1, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=0.0, disable_update=False):
    # Initialize input caches
    layers = model.blocks
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.num_prefix_tokens+model.patch_embed.num_patches, model.embed_dim), dtype=dtype, device=device
    )
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in loader:
        if cache['i'] == nsamples: break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    print('Ready.')
    for i in range(len(layers)):
        layer = layers[i]
        inps, outs = inps.to(device), outs.to(device)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(sparsity_ratio, prunen=prune_n, prunem=prune_m, percdamp=0.01, blocksize=128, disable_update=disable_update)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    torch.cuda.empty_cache()
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='vit_base_patch16_224.augreg_in1k')
    parser.add_argument("--data-path", type=str, default='/data/ilsvrc12_raw_torch')
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to model checkpoint. If not provided, a pre-trained model will be auto-downloaded.")
    parser.add_argument("--pruner", type=str, default='dense', choices=['dense', 'magnitude', 'random', 'sparsegpt', 'wanda', 'magnitude_with_budget'])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--sparse-budget", type=str, default=None)
    parser.add_argument("--enable-update", action='store_true', default=False)
    args = parser.parse_args()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create and Load model:
    latent_size = args.image_size // 8
    model = timm.create_model(args.model, pretrained=True if args.ckpt is None else False, num_classes=args.num_classes)
    if args.ckpt:
        print(f"Loading model from checkpoint: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state_dict)
    
    compression.utils.replace_linear_with_(model, compression.sparse.SparseLinearFrozen, exclude=[model.head])
    
    if args.sparse_budget is not None:
        s_b = pickle.load(open(args.sparse_budget, 'rb'))
        dyna_set_sparse_budget(model, s_b, verbose=True)
        print(f"[!] Successfully loaded sparse budget from {args.sparse_budget}")
        
        
    print(model)
    print(f"[!] Converted {args.model} to sparse model")
    model.eval() 
    model.to(device)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print(transforms)
    dataset = ImageFolder(os.path.join(args.data_path, 'val'), transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )

    if args.pruner == 'dense':
        pass
    elif args.pruner == 'magnitude':
        prune_magnitude(model, device=device, prune_n=2, prune_m=4)
    elif args.pruner == 'magnitude_with_budget':
        assert args.sparse_budget is not None, f"--sparse-budget must be provided for magnitude_with_budget"
        s_b = pickle.load(open(args.sparse_budget, 'rb'))
        prune_magnitude_with_budget(model, sparse_budget=s_b, prune_m=args.M)
    elif args.pruner == 'wanda':
        prune_wanda(model, loader=loader, nsamples=args.nsamples, batch_size=1, device=device, prune_n=args.N, prune_m=args.M)
    elif args.pruner == 'sparsegpt':
        prune_sparsegpt(model, loader=loader, nsamples=args.nsamples, batch_size=1, device=device, prune_n=args.N, prune_m=args.M, disable_update=not args.enable_update)
    else:
        raise NotImplementedError

    for name, m in model.named_modules():
        if hasattr(m, 'mask'):
            print(f"Layer {name} Sparsity: {1 - torch.sum(m.mask).item()/m.mask.numel()}")
    
    if args.save_model is not None:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")