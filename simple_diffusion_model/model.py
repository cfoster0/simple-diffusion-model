import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, reduce, repeat
from typing import Sequence, Tuple, Callable
from torch.nn import Module, ModuleList, Linear, LayerNorm, GroupNorm, Conv2d


class ConditionNHWC(Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1. / torch.logspace(-5, 5, out_features // 2)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, condition):
        freqs = torch.outer(condition, self.inv_freq) # c = d / 2
        posemb = repeat(freqs, "b c -> b (2 c)")
        odds, evens = rearrange(x, '... (j c) -> ... j c', j = 2).unbind(dim = -2)
        rotated = torch.cat((-evens, odds), dim = -1)
        return einsum("b ... d , b ... d -> b ... d", x, posemb.cos()) + einsum("b ... d , b ... d -> b ... d", rotated, posemb.sin())
    
class ConditionNCHW(Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1. / torch.logspace(-5, 5, out_features // 2)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, condition):
        freqs = torch.outer(condition, self.inv_freq) # c = d / 2
        posemb = repeat(freqs, "b c -> b (2 c)")
        odds, evens = rearrange(x, 'b (j c) ... -> b j c ...', j = 2).unbind(dim = 1)
        rotated = torch.cat((-evens, odds), dim = 1)
        return einsum("b d ... , b d ... -> b d ...", x, posemb.cos()) + einsum("b d ... , b d ... -> b d ...", rotated, posemb.sin())

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
        p = self.in_proj(x)
        q, k, v = torch.split(p, [
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

class ConditionedSequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)
        
    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.condition = ConditionNCHW(out_channels)
        self.layers = ModuleList([
            Conv2d(in_channels, out_channels, (1, 1)),
            Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
        ])
        self.norm = GroupNorm(32, out_channels)
        
    def forward(self, x, condition):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                x = x + layer(self.condition(F.gelu(self.norm(x)), condition=condition))
        return x

class BottleneckBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.condition = ConditionNHWC(channels)
        self.layers = ModuleList([SelfAttention(channels // 4, 4) for _ in range(3)])
        self.norm = LayerNorm(channels)
        
    def forward(self, x, condition):
        x = rearrange(x, "b c h w -> b h w c")
        for layer in self.layers:
            x = x + layer(self.condition(self.norm(x), condition=condition))
        x = rearrange(x, "b h w c -> b c h w")
        return x

class Bicubic(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x, *args, **kwargs):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')

class UNet(Module):
    def __init__(self, encoders_decoders: Sequence[Tuple[Module, Module]], bottleneck: Module):
        super().__init__()
        outer_pair, *inner_remaining = encoders_decoders
        self.encoder, self.decoder = outer_pair
        if inner_remaining:
            self.bottleneck = UNet(inner_remaining, bottleneck)
        else:
            self.bottleneck = bottleneck
        
    def forward(self, x, condition):
        encoded = self.encoder(x, condition=condition)
        bottlenecked = self.bottleneck(encoded, condition=condition)
        return self.decoder(torch.cat([encoded, bottlenecked], dim=1), condition=condition)
        
class Model(Module):
    def __init__(self):
        super().__init__()
        self.net = UNet([
            (ResidualBlock(3, 64), ResidualBlock(64, 3)),
            (ConditionedSequential(Bicubic(1/2), ResidualBlock(64, 128)), ConditionedSequential(ResidualBlock(128+128, 64), Bicubic(2))),
            (ConditionedSequential(Bicubic(1/2), ResidualBlock(128, 256)), ConditionedSequential(ResidualBlock(256+256, 128), Bicubic(2))),
            (ConditionedSequential(Bicubic(1/2), ResidualBlock(256, 512)), ConditionedSequential(ResidualBlock(512+512, 256), Bicubic(2))),
        ], ConditionedSequential(Bicubic(1/2), BottleneckBlock(512), Bicubic(2))
        )
       
    def forward(self, x, condition):
        return self.net(x, condition=condition)
