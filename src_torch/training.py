from autoencoder import *
import wandb
from tqdm import tqdm as tqdm_notebook

import os
import torch
import torch.nn as nn
import numpy as np

class LossLogger:
    def __init__(self, loss_names, n_losses=6):
        self.n_losses = n_losses
        self.loss_names = loss_names
        assert n_losses == len(loss_names), f"Size of loss names ({len(loss_names)}) not compatible (!= {n_losses})"
        self.reset()

    def reset(self):
        self.count = 0
        self.running_loss = 0
        self.running_losses = np.zeros((self.n_losses,))

    def log(self,loss):
        self.count += 1
        self.running_loss += loss[0].item()
        losses = np.array([loss[1][i].item() for i in range(self.n_losses)]).reshape((-1,))
        self.running_losses += losses

    def toWandb(self, name, step):
        self.running_loss /= self.count
        print(f'{name} Loss: {self.running_loss}')
        wandb.log({f"{name} Loss": self.running_loss}, step=step)

        self.running_losses /= self.count
        for i in range(self.n_losses):
            print(f'{name} {self.loss_names[i]}: {self.running_losses[i]}')
            wandb.log({f"{name} {self.loss_names[i]} Loss": self.running_losses[i]}, step=step)
        
        self.reset()
 
def train_network(AE, train_dl, test_dl, config, epochs=None):
    # AE = Autoencoder(config).to(config.device)
    if epochs is None:
        epochs = config.epochs
    optimizer = optim.Adam(AE.parameters(), lr=config.learning_rate)
    logger  = LossLogger(AE.losses_names())

    for e in range(epochs):
        AE.train()
        logger.reset()
        
        for i, data in enumerate(tqdm_notebook(train_dl,desc=f"Epoch {e}/{config.epochs}")):
            optimizer.zero_grad()
            loss = AE.loss(data)
            loss[0].backward()
            optimizer.step()
            logger.log(loss)
        logger.toWandb("Train", step=e)
        
        if config.print_progress and e % config.print_frequency == 0:
            # Evaluate
            AE.eval()
            logger.reset()
            
            with torch.no_grad():
                for i, data in enumerate(tqdm_notebook(test_dl,desc=f"Testing!")):
                    loss = AE.loss(data)
                    logger.log(loss)
            logger.toWandb("Test", step=e)