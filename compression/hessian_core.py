import torch
import torch.nn as nn
import math 


class HessianPruner():
    def __class__(self, args, name, layer):
        self.args = args
        self.name = name
        self.layer = layer
        
        self.device = layer.weight.device if hasattr(layer, 'weight') else layer.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.cols = W.shape[1]        
        self.H = torch.zeros(self.cols, self.cols, device=self.device)
        self.n_samples = 0
        
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        
        inp = inp.t()
        self.H *= self.n_samples / (self.n_samples + tmp)
        self.n_samples += tmp
        inp = math.sqrt(2 / self.n_samples) * inp.float()
        self.H += inp.matmul(inp.t())
    
    def prune(self, row_b=-1, col_b=128, sparse_n=2, sparse_m=4, percdamp:float=0.02):
        W = self.layer.weight.data.clone()
        W = W.float()
        M = torch.ones_like(W)
        
        H = self.H 
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        Losses = torch.zeros(self.rows, device=self.dev)
        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.cols, device=self.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        mask = None

        blocksize = col_b
        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            count = i2 - i1
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            M1 = M[:, i1:i2].clone()
            
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            mask1 = torch.zeros_like(W1) == 1
            
            for i in range(count):
                w = W1[:, i]
                m = M1[:, i]
                d = Hinv1[i, i]
        
                if sparse_n != 0 and i % sparse_m == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
                
                q = w.clone()
                q[mask1[:, i]] = 0
                m[mask1[:, i]] = 0

                Q1[:, i] = q
                M1[:, i] = m
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                if self.args.update_weight:
                    err1 = (w - q) / d 
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            
            if self.args.update_weight:
                W[:, i1:i2] = Q1
            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, 1) / 2
            if self.args.update_weight:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        torch.cuda.synchronize()
    
        if self.args.update_weight:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        self.layer.mask = M.to(dtype=self.layer.weight.dtype)
        return self.layer
    
    def free(self):
        torch.cuda.empty_cache()

        