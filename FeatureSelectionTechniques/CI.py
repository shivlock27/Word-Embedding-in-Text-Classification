import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as pyplot


for loopval in range(1,43):
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(loopval)+'.csv'
    df=pd.read_csv(fname,header=None)
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1:]
    
    x=x.values
    y=y.values
    
    in0=np.where(y==-1)#negative
    in0=in0[0]
    inn0=np.where(y!=-1)
    inn0=inn0[0]
    in1=np.where(y==0)#neutral
    in1=in1[0]
    inn1=np.where(y!=0)
    inn1=inn1[0]
    in2=np.where(y==1)#positive
    in2=in2[0]
    inn2=np.where(y!=1)
    inn2=inn2[0]
    
    def civalue(x):
        mx=np.mean(x)
        ci=np.zeros((2))
        ci[0]=mx-1.96*x.std()/math.sqrt(len(x))
        ci[1]=mx+1.96*x.std()/math.sqrt(len(x))
        return ci
    
    civ=np.zeros((len(x[0]),6))
    for i in range(0,x.shape[1]):
        fv=x[:,i]
        civ[i,0:2]=civalue(fv[in0])
        civ[i,2:4]=civalue(fv[in1])
        civ[i,4:6]=civalue(fv[in2])
    
    checkciv=np.zeros((len(x[0]),6))
    for i in range(0,x.shape[1]):
        fv=x[:,i]
        checkciv[i,0:2]=civalue(fv[inn0])
        checkciv[i,2:4]=civalue(fv[inn1])
        checkciv[i,4:6]=civalue(fv[inn2])
    
    count=0;
    impfeatures=[]
    for i in range(0,x.shape[1]):
        if (civ[i][0]>checkciv[i][1] or civ[i][1]<checkciv[i][0]):
            if (civ[i][2]>checkciv[i][3] or civ[i][3]<checkciv[i][2]):
                if (civ[i][4]>checkciv[i][5] or civ[i][5]<checkciv[i][4]):
                    impfeatures.append(i)
    
    finalarr=np.empty((x.shape[0],0))
    
    for i in impfeatures:
      col=x[:,i]
      col=np.reshape(col,(x.shape[0],1))
      finalarr = np.append(finalarr, col, axis=1)
    
    finalarr=np.append(finalarr,y,axis=1)
    fname='C:/Users/Shivang/Desktop/Twitter airline/'+str(42+loopval)+'.csv'
    np.savetxt(fname,finalarr, delimiter=',', fmt='%f')