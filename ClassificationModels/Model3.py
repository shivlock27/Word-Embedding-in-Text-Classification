# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:25:32 2020

@author: Shivang
"""


# Naive Bayes CLassification

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

NB_scores=[]

for i in range(127,169):
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(i)+'.csv'
    df=np.genfromtxt(fname,delimiter=',')
    classifier = GaussianNB()
    x_train, x_test, y_train, y_test = train_test_split(df[:,0:-1], df[:,-1], test_size=0.2,random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    NB_scores.append(accuracy)
    
df_final=pd.DataFrame(NB_scores)
df_final=df_final.round(3)
fname='C:/Users/Shivang/Desktop/Twitter airline/NaiveBayes_Scores2.csv'
np.savetxt(fname,df_final, delimiter=',', fmt='%f')