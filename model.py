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
        self.channel1_linear1 = nn.Linear(input_size, hidden_size)
        self.channel1_linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.channel2_gate = nn.Linear(input_size, hidden_size)
        self.channel2_linear1 = nn.Linear(input_size, hidden_size)
        self.channel2_linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.context_linear1 = nn.Linear(input_size, hidden_size)
        self.context_linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.output = nn.Linear(hidden_size, output_size)
        
        self.average_diversity = []
        self.average_loss = []
        
    def forward(self, x):
        c1 = torch.sigmoid(self.channel1_gate(x)) * self.channel1_linear1(x)
        c1 = self.channel1_linear2(torch.relu(c1))
        
        c2 = torch.sigmoid(self.channel2_gate(x)) * self.channel2_linear1(x)
        c2 = self.channel2_linear2(torch.relu(c2))
        
        context = self.context_linear1(x)
        context = self.context_linear2(torch.relu(context))
        
        x = self.output(torch.relu(c1 + c2 + context))
        x = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
        return x
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y, mask = batch
        yhat = self(x)
        opt.zero_grad()
        loss = OutputShaper.loss_fn(yhat, y, mask)
        
        # Calculate diversity loss
        diversity_diff = self.diversity_diff()
        
        diversity_loss = -diversity_diff
        
        weighted_loss = loss + diversity_loss * 0.001
        self.manual_backward(weighted_loss)
        opt.step()
        
        if batch_idx != 0:        
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('diversity_diff', diversity_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        # every 100 steps wandb log average metrics
        if batch_idx % 100 == 0:
            self.average_loss.append(loss)
            self.average_diversity.append(diversity_diff)
            wandb.log({"train_loss": torch.mean(torch.stack(self.average_loss)),
                       "diversity_diff": torch.mean(torch.stack(self.average_diversity))})
            self.average_loss = []
            self.average_diversity = []
        else:
            self.average_loss.append(loss)
            self.average_diversity.append(diversity_diff)
    
    def configure_optimizers(self):
        return SophiaG(self.parameters(), lr=5e-5, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
    

    def diversity_diff(self):
        # Calculate diversity loss between channel gates
        channel1_gate_weights = self.channel1_gate.weight
        channel2_gate_weights = self.channel2_gate.weight

        loss = torch.mean(torch.abs(torch.mm(channel1_gate_weights, channel2_gate_weights.t())))

        return loss
    
    @staticmethod
    def loss_fn(yhat, y, mask):
        masked_logits = yhat * mask.unsqueeze(1)
        masked_targets = y * mask.unsqueeze(1)

        # Compute the cross entropy loss
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')

        # Take the mean only over non-masked elements
        masked_loss = (loss * mask).sum() / mask.sum()
        
        return masked_loss
