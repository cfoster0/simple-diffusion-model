import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Module

def beta_schedule(timesteps):
    return np.linspace(1e-4, 0.02, timesteps).astype('float32')

class DiffusionWrapper(nn.Module):
    def __init__(self, net, input_shape, timesteps=1000):
        super().__init__()
        self.net = net
        self.input_shape = input_shape
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule(timesteps)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_hat_schedule = np.cumprod(self.alpha_schedule)

    @torch.no_grad()
    def generate(self, n, **kwargs):
        was_training = self.net.training
        self.net.eval()
        x = torch.randn((n,) + input_shape)
        for t in reversed(range(self.timesteps)):
            x = (self.alpha_schedule[t] ** -0.5) * (x - ((1.0 - self.alpha_schedule[t]) * (1.0 - self.alpha_schedule[t]) ** -0.5) * self.net(x, t))
            if t > 0:
                z = torch.randn((n,) + input_shape)
                x += (self.beta_schedule[t] ** 0.5) * z
        self.net.train(was_training)
        return x

    def forward(self, x, **kwargs):
        noise = torch.randn(x.shape, device=x.device)
        timestep = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        alpha_hat = torch.gather(torch.from_numpy(self.alpha_hat_schedule), 0, timestep)
        noised = einsum("b,bchw->bchw", alpha_hat.sqrt(), x) + einsum("b,bchw->bchw", (1.0 - alpha_hat).sqrt(), noise)
        predicted_noise = self.net(noised, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
        
        
