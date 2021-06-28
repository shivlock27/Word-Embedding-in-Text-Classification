# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:27:11 2020

@author: Shivang
"""

# Dataset from 1-7
# SVMSMOTE from 8-14
# SMOTE from 15-21
# BorderlineSMOTE -1 from 22-28
# BorderlineSMOTE -2 from 29-35
# ADASYN from 36-42

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
# sm = SVMSMOTE(random_state=42) 
# sm= SMOTE(random_state=42)
# sm= BorderlineSMOTE(random_state=42,kind='borderline-1')
# sm= BorderlineSMOTE(random_state=42,kind='borderline-2')
sm= ADASYN(random_state=42)
cou=0


for i in range(1,8):
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(i)+'.csv'
    df=np.genfromtxt(fname,delimiter=',')
    datan=df[:,0:-1]
    out=df[:,-1]
    X_res, y_res = sm.fit_resample(datan,out)
    y=y_res.reshape(-1,1)
    d=np.concatenate((X_res,y),axis=1)
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(35+i)+'.csv'
    np.savetxt(fname,d, delimiter=',', fmt='%f')