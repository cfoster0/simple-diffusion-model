import torch
import torch.nn.functional as F
import numpy as np

from torch import einsum
from torch.nn import Module

def beta_schedule(timesteps):
    return np.linspace(1e-4, 0.02, timesteps).astype('float32')

class DiffusionWrapper(Module):
    def __init__(self, net, input_shape, timesteps=1000):
        super().__init__()
        self.net = net
        self.input_shape = input_shape
        self.timesteps = timesteps
        self.register_buffer('beta_schedule', torch.from_numpy(beta_schedule(timesteps)))
        self.register_buffer('alpha_schedule', torch.from_numpy(1.0 - beta_schedule(timesteps)))
        self.register_buffer('alpha_hat_schedule', torch.from_numpy(np.cumprod(1.0 - beta_schedule(timesteps))))

    @torch.no_grad()
    def generate(self, n, **kwargs):
        was_training = self.net.training
        self.net.eval()
        x = torch.randn((n,) + self.input_shape).cuda()
        for t in reversed(range(self.timesteps)):
            timestep = torch.full((n,), t).cuda()
            x = (self.alpha_schedule[t] ** -0.5) * (x - ((1.0 - self.alpha_schedule[t]) * (1.0 - self.alpha_hat_schedule[t]) ** -0.5) * self.net(x, timestep))
            if t > 0:
                z = torch.randn((n,) + self.input_shape).cuda()
                x += (self.beta_schedule[t] ** 0.5) * z
        self.net.train(was_training)
        return x

    def forward(self, x, **kwargs):
        x = x.cuda()
        noise = torch.randn(x.shape).cuda()
        timestep = torch.randint(0, self.timesteps, (x.shape[0],)).cuda()
        alpha_hat = torch.gather(self.alpha_hat_schedule, 0, timestep)
        noised = einsum("b , b ... -> b ...", alpha_hat ** 0.5, x) + einsum("b , b ... -> b ...", (1.0 - alpha_hat) ** 0.5, noise)
        predicted_noise = self.net(noised, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
        
        
