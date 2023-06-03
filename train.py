import torch
from dataset import TokenLabelDataset
import os, json
import pandas as pd
import numpy as np
import tokenizers


cur_dir = os.path.dirname(os.path.realpath(__file__))

tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")

input_output_pairs = []

with open(cur_dir + '/output_shaper.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        input_output_pairs.append((json_obj['prompt'], json_obj['output']))

# tokenize text and output columns
def generate_df(input_output_pairs):
    df = pd.DataFrame(input_output_pairs, columns=['text', 'output'])
    return df

df = generate_df(input_output_pairs)

print(df.head())

# test print random row untokenized
print(df['text'].iloc[0])
print(df['output'].iloc[0])

# print 187 label 
print(df.head())

from dataset import TokenLabelDataset
from model import OutputShaper
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn, save, load, stack

# def collate_fn(batch):
#     # Separate the inputs (batches of tensors)
#     x, y = zip(*batch)

#     # Pad the first input to the maximum length
#     max_length = max([len(seq) for seq in x])
#     x_pad = []
#     for seq in x:
#         padded_seq = nn.functional.pad(seq, pad=(0, max_length - len(seq)), mode='constant', value=0)
#         x_pad.append(padded_seq)

#     # Convert the padded inputs to tensors
#     x = stack(x_pad)
#     y = stack(y)

#     return x, y

# config
epochs = 3
batch_size = 16

dataset = TokenLabelDataset(df['text'].values.tolist(), df['output'].values.tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

model = OutputShaper(tokenizer.get_vocab_size(), 64, tokenizer.get_vocab_size())
# opt = SophiaG(model.parameters(), lr=5e-5, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)

loss_fn = nn.CrossEntropyLoss()

from tqdm import tqdm
import pytorch_lightning as pl
import wandb
import deepspeed
from lightning.pytorch.loggers import WandbLogger

# Training flow
if __name__ == '__main__':
    pl.seed_everything(42)  # Set a fixed seed for reproducibility
    
    # wandb.init(project="output-shaper")
    
    wandb_logger = WandbLogger(project="output-shaper")
    trainer = pl.Trainer(
        precision='bf16',
        max_epochs=epochs,
        strategy="deepspeed_stage_3",
        logger=wandb_logger
    )

    trainer.fit(model, train_dataloaders=dataloader)

    # Save the model
    torch.save(model.state_dict(), "model.pt")

    wandb.finish()