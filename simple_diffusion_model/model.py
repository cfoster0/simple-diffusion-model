import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        self.mapping = torch.randn(mean=0, std=std, size=(out_features // 2, in_features))
       
    def forward(self, x):
        projection = self.mapping @ x
        embedding = torch.cat(torch.cos(projection), torch.sin(projection), -1)
        return embedding
        
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_to_embedding = FourierFeatures(1, dim)
       
    def forward(self, x, timestep):
        return x
