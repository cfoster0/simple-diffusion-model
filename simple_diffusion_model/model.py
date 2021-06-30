import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

class Model(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, x):
        return x
