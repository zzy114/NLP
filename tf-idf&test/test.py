from flask import Flask, abort, request, jsonify
import pandas as pd
import numpy as np  
import re
from sklearn.linear_model import LogisticRegression
from pandas import Series,DataFrame
from emotionWord import negation_words,negative_words,postive_words,reverse_words,news
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib



app = Flask(__name__)


@app.route('/get_text/', methods=['POST'])
def get_task():
    ret = excute(request.args['text'])
    return ret

def excute(text):
    #New text process
    data=[text]
    WP1 = WordsProcess(data)
    data = WP1.articles_process()
    tf1 = vectorizer.transform(data)
    tf1 = tf1.toarray()
    tfidf1=transformer.transform(tf1)
    tfidf1=tfidf1.toarray()
    tfidf1 = np.concatenate((tfidf1,WP1.created_postive_words,WP1.created_negative_words),axis=1)
    temp = vectorizer.get_feature_names()
    temp.append('created_postive_words')
    temp.append('created_negative_words')
    x=DataFrame(tfidf1)
    x.columns = temp
     
    #Prediction
    pred = log_reg.predict(x)
    pred1 = log_reg.predict_proba(x)
    
    answer = ('postive confidence:'+str(round(pred1[0][2]*100,2))+'%') if pred[0]==1 else (('negative confidence:'+str(round(pred1[0][0]*100,2))+'%') if pred[0]==-1 else ('neutral confidence:'+str(round(pred1[0][1]*100,2))+'%'))
    return answer

class WordsProcess(object):
    def __init__(self,articles,search_range=4):
        self.articles = articles.copy()
        self.search_range = search_range
        self.created_postive_words = np.zeros((len(articles),1))
        self.created_negative_words = np.zeros((len(articles),1))
        self.words = negative_words+postive_words
        
    def articles_process(self):      
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


if __name__ == "__main__":
    #extract stem of dictionary
    stemmer=SnowballStemmer('english')
    dict = sorted([stemmer.stem(t) for t in (negative_words+postive_words)])
    #Get the frame of the original model
    vectorizer=CountVectorizer(stop_words = 'english',max_features=5000) 
    tf = vectorizer.fit_transform(news) #Calculate tf
    tf = tf.toarray() #Convert a collection of text documents to a matrix of token counts 
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(tf)#Calculate tf-idf
    #Load the logistics model
    log_reg=joblib.load(r'C:\Users\zhyzhang\Desktop\log_reg.model')
    
    get_task()
    
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=8384, debug=True)





















