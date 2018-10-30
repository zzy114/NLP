# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:07:04 2018

@author: zhyzhang
"""


import numpy as np  
from pandas import Series,DataFrame
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score 


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, r'C:\Users\zhyzhang\Desktop\glove.6B')

MAX_SEQUENCE_LENGTH = 200
MAX_NUM_WORDS = 40000
EMBEDDING_DIM = 100
batch_size = 2000
n_epoch = 2000

path=r'C:\Users\zhyzhang\Desktop\News Samples\bitcoin4.csv'
df = pd.read_csv(path,index_col=None,header=0,engine='python',encoding=None)
df = df.sample(frac=1).reset_index(drop=True).dropna()
news = list(df.iloc[:,4])
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(news)
sequences = tokenizer.texts_to_sequences(news)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

label = df.iloc[:,0].reset_index()

from sklearn.cross_validation import train_test_split    
import keras
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
y_train = keras.utils.to_categorical(y_train,num_classes=3) 
y_test = keras.utils.to_categorical(y_test,num_classes=3)


#接下来，我们从GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='gb18030',errors='ignore')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation

def train_lstm(input_dim, output_dim, MAX_SEQUENCE_LENGTH, embedding_weights, x_train, y_train, x_test, y_test, batch_size, n_epoch):
    print (u'创建模型...')
    model = Sequential()
    model.add( Embedding(input_dim = input_dim,
                         output_dim=output_dim,
                         weights=[embedding_weights],
                         input_length=MAX_SEQUENCE_LENGTH,
                         trainable=False))

    model.add(LSTM(output_dim=50,
                   activation='tanh',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))

    print (u'编译模型...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    print (u"训练...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,
              validation_data=(x_test, y_test))

    print (u"评估...")
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print ('Test score:', score)
    print ('Test accuracy:', acc)
    
    return model

model = train_lstm(input_dim=len(word_index) + 1,output_dim=EMBEDDING_DIM,
            MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
            embedding_weights=embedding_matrix,
            x_train=x_train,y_train= y_train,x_test= x_test,y_test= y_test,
            batch_size=batch_size, n_epoch=n_epoch)

model.save(r'C:\Users\zhyzhang\Desktop\zzy\model.h5')


        

