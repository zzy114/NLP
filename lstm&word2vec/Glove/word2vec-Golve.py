# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:07:04 2018

@author: zhyzhang
"""


import numpy as np  
from pandas import Series,DataFrame
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, r'C:\Users\zhyzhang\Desktop\glove.6B')

MAX_SEQUENCE_LENGTH = 140
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 1000
n_epoch = 5


path=r'C:\Users\zhyzhang\Desktop\News Samples\trainingandtestdata\training.1600000.processed.noemoticon.csv'
df=pd.read_csv(path,index_col=None,header=None,engine='python',encoding=None)
df = df.sample(frac=1).reset_index(drop=True)
news = list(df.iloc[:,5])
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(news)
sequences = tokenizer.texts_to_sequences(news)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
labels = df[0]
labels = labels.replace(4,1)
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

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
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print (u'编译模型...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
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
            x_train=x_train,y_train= y_train,x_test= x_val,y_test= y_val,
            batch_size=batch_size, n_epoch=n_epoch)

path1=r'C:\Users\zhyzhang\Desktop\News Samples\trainingandtestdata\testdata.manual.2009.06.14.csv'
df1=pd.read_csv(path1,index_col=None,header=None,engine='python',encoding=None,chunksize=None)
news0 = list(df1.iloc[:,5])
#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
#tokenizer.fit_on_texts(news0)
sequences1 = tokenizer.texts_to_sequences(news0)

data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
y = df1[0]
pred = model.predict(data1)

model.save(r'C:\Users\zhyzhang\Desktop\model.h5')
tokenizer.save(r'C:\Users\zhyzhang\Desktop\tokenizer.h5')
model.save


best=0
a = 0.1
for i in range(90):
    a+=0.01
    b = a+0.02
    for j in range(78):
        if b>=1:
            break
        p = []
        for x in pred:
            if x[0]>=b:
                p.append(4)
            elif x[0]<=a:
                p.append(0)
            else:
                p.append(2)
       # print(a,b,p)
        score = accuracy_score(y, p)
        print(a,b,score)
        if score>best:
            best = score
            up = b
            down = a
        b+=0.01
