# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:53:00 2018

@author: zhyzhang
"""
# -*- coding: utf-8 -*-
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from emotionWord import negation_words,negative_words,postive_words,reverse_words
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib

#To extract stem and search for reverse words, nagation words
class WordsProcess(object):
    def __init__(self,articles,search_range=4):
        self.articles = articles.copy()
        self.search_range = search_range
        self.created_postive_words = np.zeros((len(articles),1))
        self.created_negative_words = np.zeros((len(articles),1))
        self.words = negative_words+postive_words
        
    def articles_process(self):
        #extract stem of dictionary
        stemmer=SnowballStemmer('english')
        dict = sorted([stemmer.stem(t) for t in self.words])
        #search for reverse words, nagation words
        for i,element in enumerate(self.articles):
            temp = re.sub(r"[^A-Za-z ']","",element).lower().split(' ')
            a,b = 0,0
            
            copy_temp = temp.copy()
            stem = sorted([stemmer.stem(t) for t in copy_temp])
            
            for m,n in enumerate(temp):
                #match reverse_words and negation_words
                if n in negation_words or n in reverse_words:
                    for x in range(self.search_range):
                        if m+x+1<len(temp):
                            if temp[m+x+1] in postive_words and temp[m+x+1] in copy_temp:
                                copy_temp.remove(temp[m+x+1])
                                if n in self.words and n in copy_temp:
                                    copy_temp.remove(n)
                                a+=1
                                self.created_negative_words[i,0] = a
                                break
                            elif temp[m+x+1] in negative_words and temp[m+x+1] in copy_temp:
                                copy_temp.remove(temp[m+x+1])
                                if n in self.words and n in copy_temp:
                                    copy_temp.remove(n)
                                b+=1
                                self.created_postive_words[i,0] = b
                                break
                
                #remove words not in dictionary
                n0 = stemmer.stem(n)
                if n0 not in dict and n in stem:
                    stem.remove(n0)       
            self.articles[i] = " ".join(stem)
        return self.articles

if __name__ == '__main__':
    # get tf_idf vector
    path=r'C:\Users\zhyzhang\Desktop\News Samples\bitcoin4.csv'
    df=pd.read_csv(path,index_col=None,header=0,engine='python',encoding=None,chunksize=None).iloc[:,[3,4]] #load data
    df = df.sample(frac=1).reset_index(drop=True).dropna()
    news = list(df.iloc[:,1])
    WP = WordsProcess(news)
    news1 = WP.articles_process()
    vectorizer=CountVectorizer(stop_words = 'english',max_features=5000) 
    tf = vectorizer.fit_transform(news1) #Calculate tf
    tf = tf.toarray() #Convert a collection of text documents to a matrix of token counts
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(tf)#Calculate tf-idf
    tfidf=tfidf.toarray()
    
    #Add the feature of the number of reversion
    tfidf = np.concatenate((tfidf,WP.created_postive_words,WP.created_negative_words),axis=1)
    temp = vectorizer.get_feature_names()
    temp.append('created_postive_words')
    temp.append('created_negative_words')
    x=DataFrame(tfidf)
    x.columns = temp
 
    label = df.iloc[:,0].reset_index()
    
    #spilt train data and test data
    from sklearn.cross_validation import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.2)
    
    #save train data, then resample to balance label catagories
    x_train.to_csv(r'C:\Users\zhyzhang\Desktop\1.csv',header = False,sep = ',')
    DataFrame(y_train).to_csv(r'C:\Users\zhyzhang\Desktop\2.csv',header = False,sep = ',')
    #load processed data
    path2=r'C:\Users\zhyzhang\Desktop\3.csv'
    a=pd.read_csv(path2,index_col=None,header=0,engine='python',encoding=None,chunksize=None)
    y_train = a.iloc[:,0]
    x_train = a.iloc[:,1:]
    
    #predict
    log_reg = LogisticRegression(penalty='l2',C=1,random_state=1)
    log_reg = OneVsRestClassifier(log_reg) #Multi-classifier
    log_reg.fit(x_train, y_train)
    pred = log_reg.predict(x_test)
    pred1 = log_reg.predict_proba(x_test)
    #score
    acc_score = accuracy_score(y_test, pred) 
   
    #save model
    joblib.dump(log_reg,r'C:\Users\zhyzhang\Desktop\log_reg.model')

