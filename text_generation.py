#!/usr/bin/env python
# coding: utf-8

# In[49]:


#import dependencies
import numpy
import sys
import nltk
import tensorflow as tf
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model


# In[50]:


#load data
file = open("frankenstein-2.txt", encoding='utf-8').read()


# In[51]:


#tokenization
#standardization
def tokensize_words(input):
    input = input.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return ' '.join(filtered)
#preprocess the input data
processed_inputs = tokensize_words(file)


# In[52]:


#chars to numbers
#convert characters in our input to numbers
#we'll sort the list of the set of all characters that appear in out i/p textand then use the enumerate in to get numbers
#that represent the characters
#we'll then create a dictionary that stores the keys and values, or the characters and the numbers that represent them
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i, c in enumerate(chars))


# In[53]:


#check if words to chars or chars to num(?!) has worked?
#just so we get an idea of whether our process of converting words to characters has worked
#we print the length of our variables
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters: ", input_len)
print("Total vocab: ", vocab_len)


# In[54]:


#seq length
# wer're defining how long we want an individual sequence here
#an individual sequence is a complete mapping of input characters as integers
seq_length = 100
x_data = []
y_data = []


# In[55]:


#loop through the sequence
#here we're going through the entire list of i/ps and converting the chars to numbers with a for loop
#this will create a bunch of sequence where each sequence starts with the next character in the i/p data
#beginning with the first character
for i in range(0, input_len - seq_length, 1):
    #define i/p and o/p sequence
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append([char_to_num[out_seq]])

n_patterns = len(x_data)
print("Total Patterns: ", n_patterns)


# In[56]:


#convert input to np array
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)


# In[57]:


#one-hot encoding our label data
y = to_categorical(y_data)


# In[58]:


#creating the model
#creating a sequential model
#dropout is used to prevent overfitting
model = tf.keras.Sequential()
model.add(layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(128))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(y.shape[1], activation='softmax'))


# In[59]:


#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[60]:


#saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose = 1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]


# In[61]:


#fit model and let it train
model.fit(X,y, epochs=4, batch_size=256, callbacks=desired_callbacks)


# In[62]:


#recompile model with the saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[63]:


#output of the model back into chaarcters
num_to_char = dict((i,c) for i,c in enumerate(chars))


# In[64]:


#random seed to help generate 
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print('Random Seed: ')
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


# In[65]:


#genrate the text
for i in range(1000):
    x = numpy.reshape(pattern, (1,len(pattern), 1))
    x = x/float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]


# In[ ]:




