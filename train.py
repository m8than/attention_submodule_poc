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
from sophia import SophiaG

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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = OutputShaper(tokenizer.get_vocab_size(), 256, tokenizer.get_vocab_size())
model.to('cpu')

opt = SophiaG(model.parameters(), lr=5e-5, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
loss_fn = nn.CrossEntropyLoss()

from tqdm import tqdm
import wandb
# Training flow
if __name__ == '__main__':
    wandb.init(project='output-shaper')
    for epoch in range(epochs):
        batch_progress = tqdm(dataloader, desc=f'Epoch {epoch}', position=1)
        
        for batch in dataloader:
            opt.zero_grad()
            x, y, mask = batch
            yhat = model(x)
            loss = OutputShaper.loss_fn(yhat, y, mask)
            loss.backward()
            opt.step()
            
            if batch_progress.n % 5 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "step": batch_progress.n
                })
            
            batch_progress.update(1)
            batch_progress.set_postfix({'loss': loss.item()})
    
        # print summary
    batch_progress.close()
    
    with open(cur_dir + '/model.pt', 'wb') as f:
        save(model.state_dict(), f)
    wandb.finish()