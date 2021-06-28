# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:24:57 2020

@author: Shivang
"""


# Logistic Regression 

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as pyplot


logistic_scores=[]

for i in range(85,127):
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(i)+'.csv'
    df=np.genfromtxt(fname,delimiter=',')
    logisticRegr = LogisticRegression(random_state=4,max_iter=4000)
    x_train, x_test, y_train, y_test = train_test_split(df[:,0:-1], df[:,-1], test_size=0.2,random_state=0)
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)
    logistic_scores.append(score)
    
df_final=pd.DataFrame(logistic_scores)
df_final=df_final.round(3)
fname='C:/Users/Shivang/Desktop/Twitter airline/Logistic_Scores2.csv'
np.savetxt(fname,df_final, delimiter=',', fmt='%f')