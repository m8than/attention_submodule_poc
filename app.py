from collections import Counter
import json
import os
import numpy as np
import tensorflow as tf
import tokenizers
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

cur_dir = os.path.dirname(os.path.realpath(__file__))

input_output_pairs = []

with open(cur_dir + '/test.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        input_output_pairs.append((json_obj['prompt'], json_obj['output']))

tokenizer = tokenizers.Tokenizer.from_file(cur_dir + "/20B_tokenizer.json")

# print vocab size
vocab_size = tokenizer.get_vocab_size()
print("Vocab size: {}".format(vocab_size))

def getModel(from_path=None, ctx_len=100000, embedding_dim=128, vocab_size=50281):
    if from_path is not None:
        return keras.models.load_model(from_path)
    
    model = Sequential()
    # mask 0's
    model.add(Masking(mask_value=0, input_shape=(ctx_len,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=ctx_len))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    
    model.add(Dense(vocab_size, activation='linear'))
    
    def masked_loss(y_true, y_pred):
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # mask out any 0's in the y_true
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        
        # mask out first element
        mask = mask[:, 1:]
        
        masked_loss = loss * mask
        
        return tf.reduce_mean(masked_loss)
    
    model.compile(optimizer='adam', loss=masked_loss, metrics=['accuracy'])
    return model
        
def generateInputOutputPairs(input_output_pairs):
    input_tokens = [tokenizer.encode(pair[0]).ids for pair in input_output_pairs]
    output_token_counts = [Counter(tokenizer.encode(pair[1]).ids) for pair in input_output_pairs]

    for input_sequence, output_counts in zip(input_tokens, output_token_counts):
        input_tokens = input_sequence
        output_tokens = [output_counts.get(i, 0) for i in range(vocab_size)]
        
        yield (input_tokens, output_tokens)
        
def getXYTrainTest(input_output_pairs, train_size=0.8):
    x = [input_output_pair[0] for input_output_pair in input_output_pairs]
    y = [input_output_pair[1] for input_output_pair in input_output_pairs]

    count = len(x)
    train_count = int(count * train_size)

    x_train = x[:train_count]
    y_train = y[:train_count]

    x_test = x[train_count:]
    y_test = y[train_count:]
    
    return (x_train, y_train, x_test, y_test)

print("Generating input output count pairs...")

input_output_pairs = input_output_pairs[:10] # testing model limit

input_output_pairs = list(generateInputOutputPairs(input_output_pairs))

print("Getting train test split...")
x_train, y_train, x_test, y_test = getXYTrainTest(input_output_pairs, train_size=0.8)

ctx_len = 100000

x_train = pad_sequences(x_train, maxlen=ctx_len, padding='post', value=0)
x_test = pad_sequences(x_test, maxlen=ctx_len, padding='post', value=0)


y_train = np.array(y_train)
y_test = np.array(y_test)


print(x_train.shape)

model = getModel(ctx_len=ctx_len, vocab_size=vocab_size)

print("Training model...")
model.fit(x_train, y_train, epochs=150, batch_size=4, verbose=1, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)