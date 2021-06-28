# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:18:04 2020

@author: Shivang
"""


# MLP Classifier

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as pyplot
from sklearn.tree import DecisionTreeClassifier

mlp_scores=[]

for i in range(42,85):
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(i)+'.csv'
    df=np.genfromtxt(fname,delimiter=',')
    classifier = MLPClassifier(random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(df[:,0:-1], df[:,-1], test_size=0.2,random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    mlp_scores.append(accuracy)
    
df_final=pd.DataFrame(mlp_scores)
df_final=df_final.round(3)
fname='C:/Users/Shivang/Desktop/Twitter airline/MLP_Scores2.csv'
np.savetxt(fname,df_final, delimiter=',', fmt='%f')