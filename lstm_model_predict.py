import pickle

import pandas as pd
import numpy as np
import seaborn as sns

from preprocess import df_to_list

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

df_test = pd.read_csv("./data/test3.csv")
print("Testing data's head: ")
print(df_test.head())

#preparing training data.
X_test = df_to_list(df_test['Text'])
print("Processed testing data: ")
print(X_test[:5])

#load tokenizer.
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#converting text into integer sequences.
X_test = tokenizer.texts_to_sequences(X_test)
#padding to prepare sequences of same length.
X_test = pad_sequences(X_test, maxlen=300, padding='post')
print("Tokenize testing data: ")
print(X_test[:5])

#load the saved model.
model = load_model('saved_model/my_model')

#predict the test data's labels.
y_pred = model.predict(X_test, batch_size=64)
print("First 5 raw prediction: ")
print(y_pred[:5])

#find indices of of max value in predicted values.
predicted_class = np.argmax(y_pred, axis=1)
print("First 5 numeric prediction: ")
print(predicted_class[:5])

#adding prediction to dataframe.
df_test['CLASS'] = predicted_class
#converting indices to meaningful names.
df_test['CLASS'] = df_test['CLASS'].apply(lambda label : 'positive' if label==0
                                                        else('neutral' if label==1
                                                                        else 'negative'))
#rename 'ID' to 'REVIEW-ID'.
df_test.rename({'ID': 'REVIEW-ID'}, axis=1, inplace=True)
#drop the review.
df_test = df_test.drop('Text', axis=1)
print("Testing data with predicted class labels: ")
print(df_test.head())

#distribution.
print(df_test['CLASS'].value_counts())

sns.displot(df_test['Class'], bins=10, height=6, aspect=1.5)

#write the content of dataframe to .txt file.
df_test.to_csv('./data/prediction_glove.csv', index=None, mode='w')
