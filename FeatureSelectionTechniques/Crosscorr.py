# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:54:47 2020

@author: Shivang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


for loopvar in range(1,43):
    paribas_data = pd.read_csv('C:/Users/Shivang/Desktop/Twitter airline/'+str(loopvar)+'.csv',header=None)
    
    y=paribas_data.iloc[:,-1]
    paribas_data=paribas_data.iloc[:,:-1]
    
    correlated_features = set()
    correlation_matrix = paribas_data.corr()
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    
    paribas_data.drop(labels=correlated_features, axis=1, inplace=True)
    final_arr=paribas_data.join(y)
    np.savetxt('C:/Users/Shivang/Desktop/Twitter airline/'+str(84+loopvar)+'.csv', final_arr ,delimiter=',')