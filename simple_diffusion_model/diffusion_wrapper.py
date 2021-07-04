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
    def __init__(self, net, timesteps=1000):
        super().__init__()
        self.net = net
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule(timesteps)

    @torch.no_grad()
    def generate(self, **kwargs):
        was_training = self.net.training
        self.net.eval()
        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        b = x.shape[0]

        timestep = torch.randint(0, self.timesteps, (b))
        noised = x
        noise = noised - x
        predicted_noise = self.net(noised, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        return noise
        
        
