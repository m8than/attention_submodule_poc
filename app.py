from collections import Counter
import copy
import json
import math
import os
import numpy as np
import tensorflow as tf
import tokenizers
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Masking, Lambda, Input, Attention, Dot, Concatenate, Activation, Bidirectional, Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# output available devices
print("Available devices:")
print(tf.config.list_physical_devices())

cur_dir = os.path.dirname(os.path.realpath(__file__))

input_output_pairs = []

with open(cur_dir + '/test-maximum.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        input_output_pairs.append((json_obj['prompt'], json_obj['output']))

tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")

# print vocab size
vocab_size = tokenizer.get_vocab_size()
print("Vocab size: {}".format(vocab_size))

def getModel(from_path=None, vocab_size=50281):
    if from_path is not None:
        return keras.models.load_model(from_path)
    
    num_splits = 100
    
    # Input layer
    input_layer = Input(shape=(vocab_size,))
    
    # Full input context learning
    context_layer = Dense(4096)(input_layer)
    context_layer = Dense(2048)(context_layer)
    context_layer = Dense(num_splits)(context_layer)
    
    # mask layer
    mask_layer = Masking(mask_value=0)(input_layer)

    # Split input into 4 parts (allow for 4 parallel dense layers) one may be bigger to account for uneven splits
    split_sizes = [vocab_size // num_splits] * (num_splits-1)
    split_sizes.append(vocab_size - sum(split_sizes))
    
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=split_sizes, axis=1))(mask_layer)
    
    # Define the split branches
    branches = []
    for i in range(len(split_sizes)):
        split_branch = Dense(64)(split_input[i])
        split_branch = Dropout(0.2)(split_branch)
        split_branch = Dense(32, kernel_regularizer=L2(0.001))(split_branch)
        split_branch = Dense(1)(split_branch)
        branches.append(split_branch)
    
    # Concatenate the dense layers
    concatenate = Concatenate(axis=1)([context_layer] + branches)
    
    post_concat_dense = Dense(512)(concatenate)
    post_concat_dense = Dense(256)(post_concat_dense)
    post_concat_dense = Dense(256, activation='tanh')(post_concat_dense)
    
    # Output layer
    output_layer = Dense(vocab_size, activation='linear')(post_concat_dense)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    opt = Adam(learning_rate=0.0001)

    model.compile(optimizer=opt, loss='huber_loss', metrics=['accuracy'])
    return model
        
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
    
        
def getXYTrainTest(input_output_pairs, train_size=0.8):
    input_output_pairs = np.array(input_output_pairs)
    count = len(input_output_pairs)
    train_count = int(count * train_size)

    x_train = input_output_pairs[:train_count, 0]
    y_train = input_output_pairs[:train_count, 1]
    x_test = input_output_pairs[train_count:, 0]
    y_test = input_output_pairs[train_count:, 1]
    
    return x_train, y_train, x_test, y_test


input_output_pairs = input_output_pairs[:2000] # testing model limit

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

print("Getting train test split...")
x_train, y_train, x_test, y_test = getXYTrainTest(input_output_pairs, train_size=0.8)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

print(x_train.shape)

model = getModel(vocab_size=vocab_size)

print("Training model...")

cur_dir = os.path.dirname(os.path.realpath(__file__))

# Create the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=cur_dir + '/checkpoints/chkpoint',
    save_weights_only=True,  # Set to True if you only want to save model weights
    save_freq='epoch',  # Save checkpoints at the end of each epoch
    monitor='val_loss',  # Metric to monitor for saving the best checkpoints
    mode='min',  # Mode for the monitored metric
    save_best_only=True  # Save only the best checkpoints based on the monitored metric
)


import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# Initialize a new W&B run
# wandb.init(
#     project="response-assist",
#     config={"bs": 12}
# )


model.fit(x_train,
          y_train,
          epochs=20,
          batch_size=6,
          verbose=1,
          callbacks=[
              checkpoint_callback,
#              WandbMetricsLogger("batch"),
#              WandbModelCheckpoint("models")
              ],
          validation_data=(x_test, y_test))

# save model
model.save(cur_dir + '/model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)