import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from torch.autograd import Function

"""
    Core Implementation of LSQ    
"""
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class Round(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class FunLSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            # w_q = w_q * alpha.abs()  # CHECKME 20260317 : apply abs
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float() 
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller -bigger 
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0) #?
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None
