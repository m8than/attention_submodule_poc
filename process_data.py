import json
import os
import tokenizers
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

cur_dir = os.path.dirname(os.path.realpath(__file__))
tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")

input_output_pairs = []
vocab_size = tokenizer.get_vocab_size()
print("Vocab size: {}".format(vocab_size))

with open(cur_dir + '/data/test-maximum.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        input_output_pairs.append((json_obj['prompt'], json_obj['output']))
        
def generateInputOutputPairs(input_output_pairs):
    for prompt, response in input_output_pairs:
        input_prompt_ids = tokenizer.encode(prompt).ids
        output_prompt_ids = tokenizer.encode(response).ids
        
        input_tokens = [0] * vocab_size
        output_tokens = [0] * vocab_size
        
        for token_id in input_prompt_ids:
            input_tokens[token_id] += 1
        
        for token_id in output_prompt_ids:
            output_tokens[token_id] += 1
        
        yield (input_tokens, output_tokens)
        
def removeStopwords(list_of_pairs_of_tokens, tokens_to_0):
    for input_tokens, output_tokens in list_of_pairs_of_tokens:
        for token_id in tokens_to_0:
            input_tokens[token_id] = 0
            output_tokens[token_id] = 0
        
        yield (input_tokens, output_tokens)
    
    

print("Generating and tokenize output count pairs...")
progress_bar = tqdm(total=len(input_output_pairs))
new_pairs = []

for item in generateInputOutputPairs(input_output_pairs):
    new_pairs.append(item)
    progress_bar.update(1)

progress_bar.close()
input_output_pairs = new_pairs

print("Removing stop words...")
nltk.download('punkt')
nltk.download('stopwords')

stopwords = stopwords.words()

# tokenize all stop words in the list
stopwords = [tokenizer.encode(word).ids[0] for word in stopwords]
stopwords = set(stopwords)

progress_bar = tqdm(total=len(input_output_pairs))
new_pairs = []

for item in removeStopwords(input_output_pairs, stopwords):
    new_pairs.append(item)
    progress_bar.update(1)

progress_bar.close()
input_output_pairs = new_pairs

print(len(input_output_pairs))

# save input_output_pairs to file
np.save(cur_dir + '/data/processed/testdata.npy', input_output_pairs)