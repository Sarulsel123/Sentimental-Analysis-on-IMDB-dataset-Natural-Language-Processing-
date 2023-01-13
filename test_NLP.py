#import libraries
import os
import re
import string
import h5py as h5
import numpy as np
import pandas as pd
import pickle as pk
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras.models import load_model
from keras import layers

#Preprocessing the Given Text File of Test Dataset by removing punctuations , splitting words by use of stop words
def preprocessing(d_file):  
    d_file = "".join(char for char in d_file if char not in string.punctuation and not char.isdigit())
    words_file = [word for word in d_file.split() if not word in stop_words]
    d_file = (" ".join(words_file)).strip()
    return d_file.lower()

#Using NLP Model Trained in the Train_NLP.py 
stop_words = stopwords.words('english')
NLP_Trained = load_model("/models/Group54_NLP_model.hdf")

# Preprocessing Test pos and Test neg File from respective paths 
t_data = []
t_targets = []

t_pos_path = "/data/aclImdb/test/pos"
t_neg_path = "/data/aclImdb/test/neg"
t_pos_file = os.listdir(t_pos_path)
t_neg_file = os.listdir(t_neg_path)

for file in t_pos_file:
    with open (os.path.join(t_pos_path, file), encoding="utf8") as file:
        data_file = file.readlines()
        t_data.append(preprocessing(str(data_file)))
        t_targets.append(1)

for file in t_neg_file:
    with open (os.path.join(t_neg_path, file), encoding="utf8") as file:
        data_file = file.readlines()
        t_data.append(preprocessing(str(data_file)))
        t_targets.append(0)

# Creating a DataFrame with Review and Target files 
t_data_frame = pd.DataFrame({"Review":t_data, "Target":t_targets})
print("====Preprocessing Test Data Completed for Testing Data====")


with open('/models/vector_tokens.pkl', 'rb') as file:
        tokenize_vector = pk.load(file)

t_word_vector = tokenize_vector.texts_to_sequences(t_data_frame["Review"]) 
t_padded_text = sequence.pad_sequences(t_word_vector, maxlen=450, padding='pre')
t_target_predict = NLP_Trained.predict(t_padded_text)

count = 0
for i in range(len(t_target_predict)):
    if round(t_target_predict[i][0]) == t_targets[i]:
        count += 1
# Prediction of Accuracy 
accuracy = count/len(t_target_predict)
print("TESTING ACCURACY FOR THE NLP MODEL : {0:2%}".format(accuracy))
