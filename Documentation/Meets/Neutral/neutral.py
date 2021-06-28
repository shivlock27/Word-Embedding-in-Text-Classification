import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import math
# import matplotlib.pyplot as pyplot 

# gets confidence interval for a classification
def civalue(x):
    mx=np.mean(x)
    ci=np.zeros((2))
    ci[0]=mx-1.96*x.std()/math.sqrt(len(x))
    ci[1]=mx+1.96*x.std()/math.sqrt(len(x))
    return ci

def calc(arr):
    x1=arr[0]
    x2=arr[1]
    y1=arr[2]
    y2=arr[3]
    if x2>=y1 and y2>=x1:
        return 1
    else:
        return 0

fname='C:/Users/Shivang/Desktop/Twitter airline/dataset.csv'
df=pd.read_csv(fname,header=0,encoding = 'unicode_escape')
dfpos=df[df.airline_sentiment=="neutral"]
in1=dfpos.index
# print(in1)
dfneg=df[df.airline_sentiment!="neutral"]
in0=dfneg.index
# print(in0)

vectorizer = CountVectorizer(ngram_range=(1,1))
x = vectorizer.fit_transform(df.text)

# print(vectorizer.vocabulary_)
# print(x.shape)

x=x.toarray()
df2=pd.DataFrame(x,index=df.index)
# print(df2.shape)
#print(x.shape)
# print(vectorizer.get_feature_names())

# civpb=np.zeros((15,4))
# for i in range(0,15):
#     fv=df2[i]
#     civpb[i,0:2]=civalue(fv[in0])
#     civpb[i,2:4]=civalue(fv[in1])
#     x=np.zeros((2,2))
#     x[0:2,0]=civpb[i,0:2]
#     x[0:2,1]=civpb[i,2:4]
#     pyplot.boxplot(x,labels=['Not Neutral','Neutral'])
#     pyplot.grid(True)
#     pyplot.xlabel('Metrics')
#     pyplot.ylabel('95%CI')
#     fna='C:/Users/Shivang/Desktop/Twitter airline/Neutral/'+str(i)+".png"
#     pyplot.savefig(fna)
#     pyplot.close()


civ=np.zeros((15209,4))
for i in range(0,15209):
    fv=df2[i]
    civ[i,0:2]=civalue(fv[in0])
    civ[i,2:4]=civalue(fv[in1])

# print(civ.shape)

impft=np.zeros((15209,1))
for i in range(0,15209):
    impft[i,0]=calc(civ[i])

# print(impft)


    


