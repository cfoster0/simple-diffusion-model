import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

class Rotary(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1. / torch.logspace(1.0, 10_000.0, out_features // 2)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, timestep):
        freqs = einsum('b , c -> b c', timestep, self.inv_freq) # c = d / 2
        posemb = repeat(freqs, "b c -> b (2 c)")
        out = x * posemb.cos()
        odds, evens = rearrange(x, '... (j c) -> ... j c', j = 2).unbind(dim = -2)
        rotated = torch.cat((-evens, odds), dim = -1)
        out += rotated * posemb.sin()
        return out

class Residual(nn.Module):
    def __init__(self, residual):
        """
        In the constructor we stash way the module that'll be called along
        the residual branch. This is just for convenience.
        """
        super(Residual, self).__init__()
        self.residual = residual

    def forward(self, x):
        return x + self.residual(x)
        
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_conditioning = Rotary(dim)
       
    def forward(self, x, timestep):
        x = self.timestep_conditioning(x, timestep)
        return x
