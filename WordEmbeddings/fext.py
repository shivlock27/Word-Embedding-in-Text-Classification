# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:24:12 2020

@author: lov
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
import re
import nltk
from scipy.stats import mannwhitneyu
from bs4 import BeautifulSoup
# from scipy.stats import mannwhitneyu

def civalue(x):
    mx=np.mean(x)
    ci=np.zeros((2))
    ci[0]=mx-1.96*x.std()/math.sqrt(len(x))
    ci[1]=mx+1.96*x.std()/math.sqrt(len(x))
    return ci


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


fname='E:/my_student/2020-2021_sem2/twitter-airline-sentiment.csv'
df=pd.read_csv(fname,header=0,encoding = 'unicode_escape')
df['text'] = df['text'].apply(clean_text)
vectorizer = CountVectorizer(ngram_range=(1,2))
x = vectorizer.fit_transform(df.text)

x=x.toarray()
print(x)
print(vectorizer.get_feature_names())


a=np.sum(x,axis=0)
in1=np.where(a>2)
x1=x[:,in1[0]]
x2 = vectorizer.fit_transform(df.airline_sentiment)

x2=x2.toarray()

pv=np.zeros((10508,3))
for i in range(0,3):
    in0=np.where(x2[:,i]==0)[0]
    in1=np.where(x2[:,i]==1)[0]
    for j in range(0,10508):
        w,pv[j,i]=mannwhitneyu(x1[in0,j],x1[in1,j])
        


