import torch, os
from model import OutputShaper
import tokenizers
import numpy as np

cur_dir = os.path.dirname(os.path.realpath(__file__))
tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")
model = OutputShaper.load_from_checkpoint(cur_dir + '/model.pt', input_size=50281, hidden_size=96, output_size=50281)

prompt = """<|im_start|>user
Repeat the word "dog"<|im_end|>
<|im_start|>assistant
"""
tokenized_prompt = tokenizer.encode(prompt).ids
tokenized_text = np.bincount(tokenized_prompt, minlength=tokenizer.get_vocab_size())
max_text = np.max(tokenized_text)
normalized_text = np.exp(tokenized_text - max_text) / np.sum(np.exp(tokenized_text - max_text))

# run inference
output = model(torch.tensor(normalized_text.astype(np.float32)).unsqueeze(0))

list_out = output.tolist()[0]

# list to dict
dict_out = dict(enumerate(list_out))

# sort dict by value
sorted_dict_out = {k: v for k, v in sorted(dict_out.items(), key=lambda item: item[1], reverse=False)}

# get top 30 prediction ids sorted by probability
sorted = list(sorted_dict_out.keys())

# decode top 10 predictions
counter = 0
for i in sorted:
    if tokenizer.decode([i]) == 'Dog' or tokenizer.decode([i]) == 'dog':
        print(tokenizer.decode([i]))
        break
    counter += 1
    
print(counter)