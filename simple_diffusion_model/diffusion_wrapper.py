import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
from typing import Sequence, Tuple, Callable
from torch.nn import Module, Linear, LayerNorm, GroupNorm


class DiffusionWrapper(nn.Module):
    def __init__(self, net, timesteps=1000):
        super().__init__()
        self.net = net
        self.timesteps = timesteps

    @torch.no_grad()
    def generate(self, **kwargs):
        was_training = self.net.training
        self.net.eval()
        self.net.train(was_training)
        return out

    def forward(self, x, timestep, **kwargs):
        return self.net(x, timestep)
