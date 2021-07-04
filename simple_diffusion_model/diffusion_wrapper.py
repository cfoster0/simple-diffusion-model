import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
from typing import Sequence, Tuple, Callable
from torch.nn import Module

def beta_schedule(timesteps):
    pass

class DiffusionWrapper(nn.Module):
    def __init__(self, net, input_shape, timesteps=1000):
        super().__init__()
        self.net = net
        self.input_shape = input_shape
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule(timesteps)

    @torch.no_grad()
    def generate(self, n, **kwargs):
        was_training = self.net.training
        self.net.eval()
        x = torch.randn((n,) + input_shape)
        for timestep in self.timesteps:
            x = x - self.net(x, timestep)
        self.net.train(was_training)
        return x

    def forward(self, x, **kwargs):
        b = x.shape[0]

        timestep = torch.randint(0, self.timesteps, (b))
        noised = x
        noise = noised - x
        predicted_noise = self.net(noised, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        return noise
        
        
