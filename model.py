from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, float32
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import wandb
from sophia import SophiaG

class OutputShaper(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(OutputShaper, self).__init__()
        self.automatic_optimization = False
        
        self.channel1_gate = nn.Linear(input_size, hidden_size)
        self.channel1_linear = nn.Linear(input_size, hidden_size)
        
        self.channel2_gate = nn.Linear(input_size, hidden_size)
        self.channel2_linear = nn.Linear(input_size, hidden_size)
        
        self.context_linear = nn.Linear(input_size, hidden_size)
        
        self.accumulate = nn.Linear(hidden_size, hidden_size)
        
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        c1 = torch.sigmoid(self.channel1_gate(x)) * self.channel1_linear(x)
        c2 = torch.sigmoid(self.channel2_gate(x)) * self.channel2_linear(x)
        
        context = self.context_linear(x)
        
        x = self.accumulate(torch.relu(c1 + c2 + context))
        
        x = self.output(x)
        x = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
        return x
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y, mask = batch
        yhat = self(x)
        opt.zero_grad()
        loss = OutputShaper.loss_fn(yhat, y, mask)
        self.manual_backward(loss)
        opt.step()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'train_loss': loss})
        return loss
    
    def configure_optimizers(self):
        return SophiaG(self.parameters(), lr=5e-5, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
    
    @staticmethod
    def loss_fn(yhat, y, mask):
        masked_logits = yhat * mask.unsqueeze(1)
        masked_targets = y * mask.unsqueeze(1)

        # Compute the cross entropy loss
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')

        # Take the mean only over non-masked elements
        masked_loss = (loss * mask).sum() / mask.sum()
        
        return masked_loss
