# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:56:52 2018

@author: zhyzhang
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:07:04 2018

@author: zhyzhang
"""

import gensim
import numpy as np  
from pandas import Series,DataFrame
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

google_model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\zhyzhang\Documents\GoogleNews-vectors-negative300.bin',binary=True)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 140
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.15
batch_size = 100
n_epoch = 1


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
print('Shape of data tensor:', data.shape)


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    try:  
        embedding_vector = google_model[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue


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



#遍历寻找三分类的最优阈值
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
