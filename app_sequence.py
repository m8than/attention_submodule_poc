from collections import Counter
import copy
import json
import os
import numpy as np
import tensorflow as tf
import tokenizers
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Masking, Lambda, Input, Attention, Dot, Concatenate, Activation, Bidirectional, Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras.backend as K

# output available devices
print("Available devices:")
print(tf.config.list_physical_devices())

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

def getModel(from_path=None, ctx_len=100000, vocab_size=50281):
    if from_path is not None:
        return keras.models.load_model(from_path)
    
    # Input layer
    input_layer = Input(shape=(ctx_len,))

    # Masking layer
    masking_layer = Masking(mask_value=0)(input_layer)
    
    # Embedding layer
    embedding_layer = Embedding(vocab_size, 200)(masking_layer)

    # LSTM layers
    lstm_layer = Bidirectional(LSTM(4096, return_sequences=True))(embedding_layer)
    lstm_layer = Dropout(0.2)(lstm_layer)

    lstm_layer = Bidirectional(LSTM(2048, return_sequences=True))(lstm_layer)
    lstm_layer = Dropout(0.2)(lstm_layer)
    
    lstm_layer = Bidirectional(LSTM(2048, return_sequences=True))(lstm_layer)
    lstm_layer = Dropout(0.2)(lstm_layer)
    
    lstm_layer = LSTM(4096)(lstm_layer)
    lstm_layer = Dropout(0.2)(lstm_layer)
    
    # Output layer
    output_layer = Dense(vocab_size, activation='linear')(lstm_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    def masked_categorical_crossentropy(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        
        # mask = copy of y_true with 0s replaced by 0s and anything else replaced by 1s
        mask = copy.copy(y_true)
        mask = K.cast(mask, 'bool')

        # Apply the mask to y_true and y_pred
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        # Calculate the cross-entropy loss
        loss = K.categorical_crossentropy(y_true_masked, y_pred_masked)
        
        return loss

    model.compile(optimizer='adam', loss=masked_categorical_crossentropy, metrics=['accuracy'])
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

input_output_pairs = input_output_pairs[:1000] # testing model limit

input_output_pairs = list(generateInputOutputPairs(input_output_pairs))

print("Getting train test split...")
x_train, y_train, x_test, y_test = getXYTrainTest(input_output_pairs, train_size=0.8)

ctx_len = 20000

x_train = pad_sequences(x_train, maxlen=ctx_len, padding='post', value=0)
x_test = pad_sequences(x_test, maxlen=ctx_len, padding='post', value=0)

y_train = np.array(y_train)
y_test = np.array(y_test)

print(x_train.shape)

model = getModel(ctx_len=ctx_len, vocab_size=vocab_size)

print("Training model...")
model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)