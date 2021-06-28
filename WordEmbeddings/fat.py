from gensim.models import KeyedVectors
en_model = KeyedVectors.load_word2vec_format('C:/Users/Shivang/Desktop/Twitter airline/wiki.en.vec')
import logging
import pandas as pd
import numpy as np
import gensim
import nltk
import re
from bs4 import BeautifulSoup
import csv

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

a=en_model['king']





def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    
    return np.asarray(sent_vec) / numw

# import numpy as np
# fn=['JDT','PDE','CDT','Thunderbird','Bugzilla','Platform']
# for i in range(0,6):
fname='C:/Users/Shivang/Desktop/Twitter airline/dataset.csv'
df = pd.read_csv(fname,encoding='latin-1')
df['text'] = df['text'].apply(clean_text)
print(df.head(10))
d=[];
for sent in df["text"]:
    d.append(sent_vectorizer(sent, en_model))
df1 = pd.DataFrame(d)
fname='C:/Users/Shivang/Desktop/Twitter airline/fat_dataset.csv'
df1.to_csv(fname, index=False)





#df1 = pd.DataFrame(d)    
#df1.to_csv('D:/data/msr2013-bug_dataset-master/fatThunderbird.csv', index=False)