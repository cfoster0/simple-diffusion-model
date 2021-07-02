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

class Sum(Module):
    def __init__(self):
        super().__init__()

    def forward(self, *summands):
        return sum(summands)

class Cat(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, *items):
        return torch.cat(items, dim=self.dim)

class Graph(Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def forward(self, x):
        return x, self.mapping(x)

class Residual(Module):
    def __init__(self, residual):
        """
        In the constructor we stash way the module that'll be called along
        the residual branch. This is just for convenience.
        """
        super().__init__()
        self.net = Sum(Graph(residual))

    def forward(self, x):
        return self.net(x)

class SelfAttention(Module):
    def __init__(self, head_dim: int, heads: int):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.hidden_dim = head_dim * heads
        self.in_proj = nn.Linear(head_dim * heads, head_dim * heads * 3)
        self.out_proj = nn.Linear(head_dim * heads, head_dim * heads)
        
    def forward(self, x):
        b, h, w, d = x.shape
        x = rearrange(x, "b h w d -> b (h w) d")
        q, k, v = torch.split(x, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b i h d", h=self.heads), (q, k, v))
        a = einsum("b i h d, b j h d -> b h i j", q, k) * (self.head_dim ** -0.5)
        a = F.softmax(a, dim=-1)
        o = einsum("b h i j, b j h d -> b i h d", a, v)
        o = rearrange(o, "b i h d -> b i (h d)")
        x = self.out_proj(o)
        x = rearrange(x, "b (h w) d -> b h w d", h=h, w=w)
        return x

class UNet(Module):
    def __init__(self, encdec_pairs: Sequence[Tuple[Module, Module]], bottleneck: Module):
        super().__init__()
        outer, *inner = encdec_pairs
        enc, dec = outer
        if inner:
            self.net = nn.Sequential(enc, Cat(Graph(UNet(inner, bottleneck))), dec)
        else:
            self.net = nn.Sequential(enc, Cat(Graph(bottleneck)), dec)
        
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
