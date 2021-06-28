import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import math

# gets confidence interval for a classification
def civalue(x):
    mx=np.mean(x)
    ci=np.zeros((2))
    ci[0]=mx-1.96*x.std()/math.sqrt(len(x))
    ci[1]=mx+1.96*x.std()/math.sqrt(len(x))
    return ci

fname='C:/Users/Shivang/Desktop/Twitter airline/dataset.csv'
df=pd.read_csv(fname,header=0,encoding = 'unicode_escape')
dfpos=df[df.airline_sentiment=="positive"]
in1=dfpos.index
# print(in1)
dfneg=df[df.airline_sentiment!="positive"]
in0=dfneg.index
# print(in0)

vectorizer = CountVectorizer(ngram_range=(1,2))
x = vectorizer.fit_transform(df.airline)
x=x.toarray()
print(x.shape)
print(vectorizer.get_feature_names())
civ=np.zeros((11,4))
for i in range(0,10):
	fv=x[:,i]
	civ[i,0:2]=civalue(fv[in0])
	civ[i,2:4]=civalue(fv[in1])
	x=np.zeros((11,2))
	x[i:2,0]=civ[i,0:2]
	x[i:2,1]=civ[i,2:4]
	print(x)
# fv=df[df.columns[5]]
# # print(in0)
# civ=np.zeros((2,4))
# civ[0,0:2]=civalue(fv[in0])
# civ[0,2:4]=civalue(fv[in1])
# x=np.zeros((2,2))
# x[0:2,0]=civ[0,0:2]
# x[0:2,1]=civ[0,2:4]
# print(x)

# for i in range(0,15):
#     fv=df.get_loc(i)
#     print(in0)
#     civ[i,0:2]=civalue(fv[in0])
#     civ[i,2:4]=civalue(fv[in1])
#     x=np.zeros((2,2))
#     x[0:2,0]=civ[i,0:2]
#     x[0:2,1]=civ[i,2:4]
#     print(x)
#     pyplot.boxplot(x,labels=['Not-Job','JOB'])
#     pyplot.grid(True)
#     pyplot.xlabel('Metrics')
#     pyplot.ylabel('95%CI')
#     fna='C:/Users/lov/Documents/dsv/'+str(i)+".png"
#     pyplot.savefig(fna)
#     pyplot.close()
