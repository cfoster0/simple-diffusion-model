# This code was adapted from lucidrains existing `x-transformers` repository.
from simple_diffusion_model import Model
from simple_diffusion_model import DiffusionWrapper

import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from torchvision.datasets import STL10
from torch.utils.data import DataLoader

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def scale(x):
    return x * 2 - 1

def rescale(x):
    return (x + 1) / 2

def train():
    wandb.init(project="simple-diffusion-model")

    model = DiffusionWrapper(Model(), input_shape=(3, 32, 32))
    model.cuda()

    train_dataset = CIFAR10(root='./data', split='train', transform=transforms.ToTensor(), download=True)
    val_dataset = CIFAR10(root='./data', split='test', transform=transforms.ToTensor(), download=True)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        start_time = time.time()
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            batch, _ = next(train_loader)
            loss = model(scale(batch))
            loss.backward()

        end_time = time.time()
        print(f'training loss: {loss.item()}')
        train_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()


        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                batch, _ = next(val_loader)
                loss = model(scale(batch))
                print(f'validation loss: {loss.item()}')
                val_loss = loss.item()

        if i % GENERATE_EVERY == 0:
            model.eval()
            samples = model.generate(1)
            image_array = rescale(samples)
            images = wandb.Image(image_array, caption="Generated")
            wandb.log({"examples": images}, commit=False)
        
        logs = {}
        
        logs = {
          **logs,
          'iter': i,
          'step_time': end_time - start_time,
          'train_loss': train_loss,
          'val_loss': val_loss,
        }
        
        wandb.log(logs)
      
    wandb.finish()

if __name__ == '__main__':
    train()
