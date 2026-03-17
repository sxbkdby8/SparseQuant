import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from tqdm import tqdm   
import gc

def _is_target_layer(name, module):
    return isinstance(module, (nn.Linear, )) and any(t in name for t in ['qkv', 'proj', 'fc1', 'fc2'])        

def check_memory(message=""):
    print(f"{message} - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"{message} - GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"{message} - GPU max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def normalization(v):
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def get_params_grad(model: nn.Module):
    params = []
    grads = []
    for name, m in model.named_modules():
        if not _is_target_layer(name, m) or not hasattr(m, 'weight') or not m.weight.requires_grad:
            print(f"Skipping {name} ")
            continue
        param = m.weight
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def get_layer_params_grad(model: nn.Module):
    layer_params = {}
    layer_grads = {}
    
    for name, m in model.named_modules():
        if not _is_target_layer(name, m) or not hasattr(m, 'weight') or not m.weight.requires_grad:
            continue
            
        param = m.weight
        grad = 0. if param.grad is None else param.grad + 0.
        
        layer_params[name] = [param]
        layer_grads[name] = [grad]
    
    return layer_params, layer_grads

def hessian_vector_product(gradsH, params, v):
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv

def orthnormal(w, v_list):
    """Make vector w orthogonal to each vector in v_list, then normalize"""
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

def group_add(xs, ys, alpha=1.0):
    """Vector group addition"""
    return [x + alpha * y for x, y in zip(xs, ys)]

def move_to_cpu(tensor_dict):
    """Move tensors in dict to CPU and detach"""
    cpu_dict = {}
    for key, value in tensor_dict.items():
        if torch.is_tensor(value):
            cpu_dict[key] = value.detach().cpu()
        else:
            cpu_dict[key] = value
    return cpu_dict

class HessianAnalyzer():
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), dataloader=None, cuda=True):
        assert (dataloader is not None)
        self.cuda = cuda
        self.model = model.eval()
        self.criterion = criterion
        self.dataloader = dataloader
        self.full_dataset = True

        if cuda:
            self.device = 'cuda'
            self.model = self.model.to(self.device)
        else:
            self.device = 'cpu'
            
        self.layer_params, self.layer_grads = get_layer_params_grad(self.model)
        
        self.layer_results = defaultdict(dict)
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.cuda:
            torch.cuda.empty_cache()
    
    def layer_hv_product(self, layer_name, v, num_samples=32):
        """Compute Hessian-vector product for specific layer"""
        device = self.device
        num_data = 0
        
        layer_param = self.layer_params[layer_name][0]
        THv = torch.zeros(layer_param.size()).to(device)
        
        for idx, (inputs, targets) in enumerate(self.dataloader):
            if idx >= num_samples:
                break
                
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            
            _, current_layer_grads = get_layer_params_grad(self.model)
            layer_grad = current_layer_grads[layer_name][0]
            current_layer_grads = None  # NOTE : this line seems to be necessary for solving memory leaking problem
            
            self.model.zero_grad()
            
            # Compute HVP for this layer
            Hv = torch.autograd.grad(layer_grad,
                                   layer_param,
                                   grad_outputs=v,
                                   only_inputs=True,
                                   retain_graph=False)
            
            THv = THv + Hv[0] * float(tmp_num_data)
            num_data += float(tmp_num_data)
            
            # Clear intermediate gradients
            outputs, loss, layer_grad, Hv = None, None, None, None
            del outputs, loss, layer_grad, Hv
            gc.collect(generation=2)
            if self.cuda:
                torch.cuda.empty_cache()

        THv = THv / float(num_data)
        eigenvalue = torch.sum(THv * v).cpu().item()
        return eigenvalue, THv

    def analyze_layers_traces(self, maxIter=100, tol=1e-3, num_samples=32):
        """Analyze traces for each layer with memory optimization"""
        device = self.device
        
        for layer_name in self.layer_params.keys():
            # print(f"Analyzing trace for layer: {layer_name}")
            check_memory(message=f"Analyzing layer: {layer_name}")
            layer_result = {}
            
            trace_vhv = []
            trace = 0.
            layer_param = self.layer_params[layer_name][0]
            
            num_params = layer_param.numel() 
            
            for i in tqdm(range(maxIter), desc=f"Trace {layer_name}"):
                self.model.zero_grad()
                # Generate Rademacher random variables
                v = torch.randint_like(layer_param, high=2, device=device)
                v[v == 0] = -1
                
                _, Hv = self.layer_hv_product(layer_name, v, num_samples)
                
                # trace_value = torch.sum(Hv * v).cpu().item()
                trace_value = group_product(Hv, v).cpu().item()
                trace_vhv.append(trace_value)
                
                # Clear intermediate tensors  
                # IMPORTANT : TO AVOID GPU MEM LEAK
                del v, Hv
                self._clear_gpu_cache()
                
                if len(trace_vhv) > 1:
                    current_mean = np.mean(trace_vhv)
                    if abs(current_mean - trace) / (abs(trace) + 1e-6) < tol:
                        break
                    else:
                        trace = current_mean
                else:
                    trace = trace_vhv[0]
            
            avg_trace = np.mean(trace_vhv)
            sum_trace = np.sum(trace_vhv)
            importance = sum_trace / num_params
            
            # layer_result['trace'] = np.mean(trace_vhv)
            layer_result['trace'] = avg_trace
            layer_result['sum_trace'] = sum_trace
            layer_result['importance'] = importance
            layer_result['trace_history'] = trace_vhv
            
            # Store results
            self.layer_results[layer_name].update(layer_result)
            print(f"Layer {layer_name}: trace = {layer_result['trace']:.6f}")
            
            self._clear_gpu_cache()
        
        return self.layer_results
    
    def analyze_layers_eigenvalues(self, maxIter=100, tol=1e-3, num_samples=32):
        """Analyze eigenvalues for each layer with memory optimization"""
        device = self.device
        
        for layer_name in self.layer_params.keys():
            print(f"Analyzing eigenvalues for layer: {layer_name}")
            layer_result = {}
            
            eigenvalues = []
            eigenvectors = []
            computed_dim = 0
            top_n = 1  # Only compute largest eigenvalue
            
            while computed_dim < top_n:
                eigenvalue = None
                layer_param = self.layer_params[layer_name][0]
                v = torch.randn(layer_param.size()).to(device)
                v = normalization([v])[0]
                
                for i in range(maxIter):
                    self.model.zero_grad()
                    tmp_eigenvalue, Hv = self.layer_hv_product(layer_name, v, num_samples)
                    
                    if eigenvalue is None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                    
                    v = normalization([Hv])[0]
                    
                    # Clear intermediate tensors
                    del Hv
                    self._clear_gpu_cache()
                
                eigenvalues.append(eigenvalue)
                # Move eigenvector to CPU to save GPU memory
                eigenvectors.append(v.detach().cpu())
                computed_dim += 1
                
                # Clear iteration variables
                del v
                self._clear_gpu_cache()
            
            # Store results and move to CPU
            layer_result['eigenvalues'] = eigenvalues
            layer_result['eigenvectors'] = eigenvectors
            self.layer_results[layer_name].update(move_to_cpu(layer_result))
            
            print(f"Layer {layer_name}: max eigenvalue = {eigenvalues[0]:.6f}")
            
            # Force cleanup after each layer
            self._clear_gpu_cache()
        
        return self.layer_results
    def get_layer_eigenvalues(self, layer_name):
        """Get eigenvalues for specific layer"""
        if layer_name in self.layer_results and 'eigenvalues' in self.layer_results[layer_name]:
            return self.layer_results[layer_name]['eigenvalues']
        return None

    def get_layer_trace(self, layer_name):
        """Get trace for specific layer"""
        if layer_name in self.layer_results and 'trace' in self.layer_results[layer_name]:
            return self.layer_results[layer_name]['trace']
        return None
