# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:16:08 2020

@author: Shivang
"""

# Feature Extraction with PCA
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.decomposition import PCA

for i in range(1,43):
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(i)+'.csv'
    dataframe=np.genfromtxt(fname,delimiter=',')
    X = dataframe[:,0:-1]
    Y = dataframe[:,-1]
    Y_data=pd.DataFrame(data=Y)
    # feature extraction
    pca = PCA(n_components=100)
    dfpc=pca.fit_transform(X)
    df = pd.DataFrame(data = dfpc)
    df_final=pd.concat([df,Y_data],axis=1)
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(126+i)+'.csv'
    np.savetxt(fname,df_final, delimiter=',', fmt='%f')
    