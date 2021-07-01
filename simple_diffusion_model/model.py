import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

class RotaryFeatures(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1. / torch.logspace(1.0, 100.0, out_features // 2)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, timestep):
        freqs = einsum('n , c -> n c', timestep, self.inv_freq) # c = d / 2
        posemb = repeat(freqs, "n c -> n (2 c)")
        out = x * posemb.cos()
        odds, evens = rearrange(x, '... (j c) -> ... j c', j = 2).unbind(dim = -2)
        rotated = torch.cat((-evens, odds), dim = -1)
        out += rotated * posemb.sin()
        return out

class FourierFeatures(nn.Module):
    def __init__(self, out_features, std=1.0):
        super().__init__()
        self.mapping = torch.randn(mean=0, std=std, size=(out_features // 2, 1))
       
    def forward(self, timestep):
        projection = self.mapping @ timestep
        embedding = torch.cat(torch.cos(projection), torch.sin(projection), -1)
        return embedding
        
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_to_embedding = RotaryFeaturesFeatures(dim)
       
    def forward(self, x, timestep):
        return x
