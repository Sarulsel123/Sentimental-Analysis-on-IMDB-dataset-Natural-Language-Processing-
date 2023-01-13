#import libraries
import os
import re
import string
import numpy as np
import pandas as pd
import pickle as pk
import nltk
import tensorflow as tf
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from gensim.models import Word2Vec
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras import layers

# Preprocessing the Given Text File of Training Dataset by removing punctuations , splitting words by use of stop words
def preprocessing(d_file):  
    d_file = "".join(char for char in d_file if char not in string.punctuation and not char.isdigit())
    words_file = [word for word in d_file.split() if not word in stop_words]
    d_file = (" ".join(words_file)).strip()
    return d_file.lower()

# The Target and the train_pos and train_neg files are preprocessed and appended to the target 
data_final = []
stop_words = stopwords.words('english')
target = []
file_pos_path = '/data/aclImdb/train/pos'
file_neg_path = '/data/aclImdb/train/neg'
d_pos_file = os.listdir(file_pos_path)
d_neg_file = os.listdir(file_neg_path)
for file in d_pos_file:
    with open (os.path.join(file_pos_path,file), encoding="utf8") as file:
        data_files = file.readlines()
        data_final.append(preprocessing(str(data_files)))
        target.append(1)

for file in d_neg_file:
    with open (os.path.join(file_neg_path,file), encoding="utf8") as file:
        data_files = file.readlines()
        data_final.append(preprocessing(str(data_files)))
        target.append(0)

# A new Data frame is created which holds the review data and target which is being compared 
data_frame = pd.DataFrame({"Review":data_final, "Target":target})
data_sample_f = data_frame.sample(frac = 1)
print("====Preprocessing Completed for Training Dataset====")

# Tokeninzing and Word to vector is done which generates the pickle file 
# We create a embedded layer which acts a foundation for the neural network 
d_tokens = [word_tokenize(review) for review in data_sample_f["Review"]]
vector = Word2Vec(d_tokens, size = 400, window = 10, min_count = 8)
Word_vector = vector.wv.vectors
tokenize = text.Tokenizer(num_words = Word_vector.shape[0])
tokenize.fit_on_texts(data_sample_f["Review"])
pk.dump(tokenize,open("/models/vector_tokens.pkl", "wb"))
print("====Tokenized Vectors Saved as Pickle File in the Directory ====")
train_target= np.array(data_sample_f["Target"])
print(Word_vector.shape[0])
embedding_layer_nn = layers.Embedding(input_dim = Word_vector.shape[0], output_dim = Word_vector.shape[1],weights= [Word_vector], trainable=True, input_length=450)

# Creating neural network with the sequential flow and using BinaryCrossEntropy 
# We use Sigmoid as the activation function for our model 
nn = Sequential()
nn.add(embedding_layer_nn)
nn.add(layers.LSTM(24, return_sequences=True))
nn.add(layers.LSTM(4))
nn.add(layers.Dense(1,activation="sigmoid"))
nn.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
print(nn.summary())

# The accuracy is processed across each epoch 
nn_encoded_text = tokenize.texts_to_sequences(data_sample_f["Review"])
nn_padded_text = sequence.pad_sequences(nn_encoded_text, maxlen = 450, padding = 'pre')
nn.fit(nn_padded_text, train_target, validation_split=0.2, epochs=5)
nn.save("/models/Group54_NLP_model.hdf")
print("====Group 54_NLP Model Saved====")
