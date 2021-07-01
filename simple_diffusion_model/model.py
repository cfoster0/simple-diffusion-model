import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from typing import Sequence, Tuple, Callable
from torch.nn import Module

class Rotary(Module):
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

class Residual(Module):
    def __init__(self, residual):
        """
        In the constructor we stash way the module that'll be called along
        the residual branch. This is just for convenience.
        """
        super().__init__()
        self.residual = residual

    def forward(self, x):
        return x + self.residual(x)

class CatCall(Module):
    def __init__(self, callable: Callable, dim=-1):
        """
        This wrapper is similar to Residual. It runs a function or
        module on an input and then concatenates the result with the
        original input along a specified dimension.
        """
        super().__init__()
        self.callable = callable
        self.dim = dim

    def forward(self, x):
        return torch.cat([x, self.callable(x)], dim=self.dim)

class UNet(Module):
    def __init__(self, encdec_pairs: Sequence[Tuple[Module, Module]], bottleneck: Module):
        super().__init__()
        outer, *inner = encdec_pairs
        enc, dec = outer
        if inner:
            self.net = nn.Sequential(enc, CatCall(UNet(inner, bottleneck)), dec)
        else:
            self.net = nn.Sequential(enc, CatCall(bottleneck), dec)
        
    def forward(self, x):
        return self.net(x)
        
class Model(Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_conditioning = Rotary(dim)
        self.unet = UNet()
       
    def forward(self, x, timestep):
        x = self.timestep_conditioning(x, timestep)
        x = self.unet(x)
        return x
