from torch.utils.data import Dataset
from torch import tensor
import numpy as np

class TokenLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokenized_labels = self.tokenizer.encode(self.labels[idx]).ids
        tokenized_text = self.tokenizer.encode(self.texts[idx]).ids
        
        # convert tokenized text to counts of all ids
        tokenized_labels = np.bincount(tokenized_labels, minlength=self.tokenizer.get_vocab_size())
        tokenized_text = np.bincount(tokenized_text, minlength=self.tokenizer.get_vocab_size())
        
        # create mask from tokenized text
        mask = np.ones(tokenized_labels.shape)
        mask[tokenized_labels == 0] = 0
        
        # softmax labels while preventing overflow
        max_label = np.max(tokenized_labels)
        max_text = np.max(tokenized_text)

        normalized_labels = np.exp(tokenized_labels - max_label) / np.sum(np.exp(tokenized_labels - max_label))
        normalized_text = np.exp(tokenized_text - max_text) / np.sum(np.exp(tokenized_text - max_text))
        
        # convert to dtype bfloat16
        tokenized_labels = normalized_labels.astype(np.float32)
        tokenized_text = normalized_text.astype(np.float32)
        
        return tensor(tokenized_text), tensor(tokenized_labels), tensor(mask)