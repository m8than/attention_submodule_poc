from torch import nn, float32
import torch
from torch.nn import functional as F

class OutputShaper(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OutputShaper, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        hidden_out = self.hidden(x)
        conv_out = self.conv_net(hidden_out)
        x = conv_out
        x = self.output(x)
        x = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
        return x
    
    
    @staticmethod
    def loss_fn(yhat, y, mask):
        masked_logits = yhat * mask.unsqueeze(1)
        masked_targets = y * mask.unsqueeze(1)

        # Compute the cross entropy loss
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')

        # Take the mean only over non-masked elements
        masked_loss = (loss * mask).sum() / mask.sum()
        
        return masked_loss
