import pickle

import pandas as pd
import numpy as np

from preprocess import df_to_list

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the training data.
df_train = pd.read_csv("./data/train3.csv")

#adding positive, neutral and negative attributes (one-hot encoding of Class).
df_train['Positive'] = np.where(df_train['Class'] == 'positive', 1, 0)
df_train['Neutral'] = np.where(df_train['Class'] == 'neutral', 1, 0)
df_train['Negative'] = np.where(df_train['Class'] == 'negative', 1, 0)
print("Training data's head: ")
print(df_train.head())

#preparing training data.
X_train = df_to_list(df_train['Text'])
y_train = df_train[['Positive', 'Neutral', 'Negative']].values
print("Processed training data: ")
print(X_train[:5])

#Tokenize the sentences.
tokenizer = Tokenizer()
#preparing vocabulary.
tokenizer.fit_on_texts(X_train)
#save tokenizer to process test data.
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#converting text into integer sequences.
X_train = tokenizer.texts_to_sequences(X_train)
#length of longest sentence.
max_len = 300
#padding to prepare sequences of same length.
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
print("Tokenize training data: ")
print(X_train[:5])

#+1 for padding.
vocab_size = len(tokenizer.word_index) + 1
print("Vocab size: %s" % vocab_size)

#load the GloVe pretrained model.
print("Loading pre-trained word-embedding model...")
embeddings_dict = {}
with open('glove.6B.300d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector
print("Loaded %s word vectors." % len(embeddings_dict))

#create a weight matrix for words in training docs.
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#hyperparameters.
units = 500
dropout = 0.5
batch_size = 64

#the model.
model = Sequential()
#embedding layer.
model.add(Embedding(vocab_size,
                    300,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))
#lstm layer.
model.add(LSTM(units, return_sequences=True, dropout=dropout))
#global maxpooling.
model.add(GlobalMaxPooling1D())
#dense layers.
model.add(Dense(units, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(units, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(units, activation='relu'))
#output layer.
model.add(Dense(3, activation='softmax'))
#compile model.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#model's summary.
print(model.summary())

print("Training the LSTM...")
#train the model.
history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=1)
print("Training completed")

#save the model.
model.save('saved_model/my_model')
