import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from einops import rearrange, reduce, repeat
from typing import Sequence, Tuple, Callable
from torch.nn import Module, Linear, Sequential, Identity


class Rotary(Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1. / torch.logspace(1.0, 10_000.0, out_features // 2)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, timestep):
        freqs = einsum('b , c -> b c', timestep, self.inv_freq) # c = d / 2
        posemb = repeat(freqs, "b c -> b (2 c)")
        odds, evens = rearrange(x, '... (j c) -> ... j c', j = 2).unbind(dim = -2)
        rotated = torch.cat((-evens, odds), dim = -1)
        out = (x * posemb.cos()) + (rotated * posemb.sin())
        return out

class SelfAttention(Module):
    def __init__(self, head_dim: int, heads: int):
        super().__init__()
        hidden_dim = head_dim * heads
        self.head_dim = head_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.in_proj = Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        
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

class ResidualBlock(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = Identity()
        
    def forward(self, x, timestep):
        return x + self.net(x)

class EncoderBlock(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layers = ModuleList([ResidualBlock(dim) for _ in range(3)])
        
    def forward(self, x, timestep):
        for layer in self.layers:
            x = layer(x, timestep)
        return x

class DecoderBlock(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layers = ModuleList([ResidualBlock(dim) for _ in range(3)])
        
    def forward(self, x, timestep):
        for layer in self.layers:
            x = layer(x, timestep)
        return x

class BottleneckBlock(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embed_timestep = Rotary(dim)
        self.layers = ModuleList([SelfAttention(dim // 4, 4) for _ in range(3)])
        
    def forward(self, x, timestep):
        for layer in self.layers:
            x = layer(self.embed_timestep(x, timestep))
        return x

class Bicubic(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x, timestep):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic'), timestep

class UNet(Module):
    def __init__(self, encdec_pairs: Sequence[Tuple[Module, Module]], bottleneck: Module):
        super().__init__()
        outer, *inner = encdec_pairs
        encoder, decoder = outer
        self.encoder = encoder
        self.decoder = decoder
        if inner:
            self.detour = UNet(inner, bottleneck)
        else:
            self.detour = bottleneck
        
    def forward(self, x, timestep):
        encoded = self.encoder(x, timestep)
        detoured = self.detour(encoded, timestep)
        decoded = self.decoder(torch.cat([encoded, detoured], dim=-1), timestep)
        return decoded
        
class Model(Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet([
            (EncoderBlock(64), DecoderBlock(64)),
            (Sequential(Bicubic(0.5), EncoderBlock(128)), Sequential(DecoderBlock(128), Bicubic(2))),
            (Sequential(Bicubic(0.5), EncoderBlock(256)), Sequential(DecoderBlock(256), Bicubic(2))),
            (Sequential(Bicubic(0.5), EncoderBlock(512)), Sequential(DecoderBlock(512), Bicubic(2))),
        ], Sequential(Bicubic(0.5), BottleneckBlock(1024), Bicubic(2.0))
       
    def forward(self, x, timestep):
        x = self.unet(x, timestep)
        return x
