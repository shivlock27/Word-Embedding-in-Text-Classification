# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:05:47 2020

@author: lov
"""

import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import torch
import transformers as ppb 


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

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
# fn=['JDT','PDE','CDT','Thunderbird','Bugzilla','Platform']
# for k in range(0,6):
fname='C:/Users/Shivang/Desktop/Twitter airline/dataset.csv'
df = pd.read_csv(fname,encoding='latin-1')
df['text'] = df['text'].apply(clean_text)
print(df.head(10))
x = tfidf.fit_transform(df['text'])
df_tfidf = pd.DataFrame(x.toarray())
x=np.array(df_tfidf)
fname='C:/Users/Shivang/Desktop/Twitter airline/tfidf_dataset.csv'
np.savetxt(fname,x, delimiter=',', fmt='%f')  
 



noc=np.zeros((x.shape[1]))
for i in range(0,x.shape[1]):
    y=x[:,i];
    in1=np.where(y!=0)
    noc[i]=len(in1)
    
    