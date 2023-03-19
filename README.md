# Sentimental-Analysis-on-IMDB-dataset-Natural-Language-Processing-


# IMDB DATASET: 
This dataset contains a lot more data than earlier benchmark datasets for binary sentiment categorization. For training and testing, it offer sets of 25,000 highly polar movie evaluations. Additional unlabeled data is also available for use. There are two types available: raw text and already processed bag of words. For more information, refer to the README file in the release.

# The sentimental analysis is performed in the following ways: 

# PREPROCESSING
The Prepocessing() is the function used in the code.
We started by erasing punctuation and digits one by one from the supplied files. 
The stop words were then eliminated from the data using the nltk corpus' stop words package. 
The data is then added to a single data frame after all the words have been combined. 
Positive and negative review data are labelled with "1" and "0," respectively, and are kept as targets in the data frame.

# TOKENIZATION
Data are initially transformed into tokens using a tokenizer from the Keras package after processing. 
Using the gensim word2vec programme, these tokens are transformed into vectors for use in training the model. I set the word2vec model's ideal dimension size at 400 and the required number of word repeats at 8. 
The test data is used to convert the embedded vector data into a similar vector form by saving it as a pickle file.


# DESINGING OF NETWORK 
The neural network has three layers: an input layer, an LSTM layer with a size of 24, another LSTM layer with a size of 4, and a single neuron in the final output layer.
This model's output will either be "1" (a favourable evaluation) or "0." (negative comment). 
The optimizer being utilised is "Adam," which thanks to its memory efficiency and superior processing performs better for huge datasets. 
Given that the output falls between [0,1], the "Sigmoid" activation function is utilised. 
The LSTM is evaluated using various numbers of neurons with various numbers of hidden layers (e.g., 2, 3, 4). Underfitting results from adding more layers, and overfitting results from adding too few or too many neurons. The model was trained using a default learning rate over 5 epochs.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/122580255/226148728-bc481100-a099-40b5-b7c7-47810479ce8a.png">

# ACCURACY OF TRAINING DATASET
After 5 iterations, the model's training accuracy is 98.9% and its validation accuracy is 86.7 percent. 
The training takes about 7 minutes every epoch, which is a reasonable length of time. 
The model's training consumes a lot of memory. The memory use is higher and the execution is interrupted due to memory exhaustion while testing multiple strategies with various numbers of neurons in the hidden layers.

 <img width="496" alt="image" src="https://user-images.githubusercontent.com/122580255/226148739-d5343fe7-4d38-44cc-b2af-9ec27eabcf96.png">

# ACCURACY OF TESTING DATASET
After preprocessing the test data and using the trained NLP model we achieve accuracy of 84.30%
 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/122580255/226148747-e7546cb2-98b1-410d-b0f3-6369302c729c.png">
