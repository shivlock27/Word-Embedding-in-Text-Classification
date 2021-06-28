import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import math
# import matplotlib.pyplot as pyplot 
import re
import nltk
# from scipy.stats import mannwhitneyu
from bs4 import BeautifulSoup

# gets confidence interval for a classification
def civalue(x):
    mx=np.mean(x)
    ci=np.zeros((2))
    ci[0]=mx-1.96*x.std()/math.sqrt(len(x))
    ci[1]=mx+1.96*x.std()/math.sqrt(len(x))
    return ci

# checks for relevant features using overlap of confidence intervals
def calc(arr):
    x1=arr[0]
    x2=arr[1]
    y1=arr[2]
    y2=arr[3]
    if x2>=y1 and y2>=x1:
        return 1
    else:
        return 0

def crosscorr(x,n):
    sef=np.zeros((n))
    for i in range(0,n):
        sef[i]=i+1
    for i in range(0,x.shape[1]-1):
        for j in range(0,x.shape[1]-1):
            if (np.array(np.where(sef==i)).shape[1]>0 and np.array(np.where(sef==j)).shape[1]>0):
                if i!=j and (x[i,j]>=0.7 or x[i,j]<=-0.7):
                    sef[np.array(np.where(sef==j))]=0;       
    return sef 

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

     
fname='C:/Users/Shivang/Desktop/Twitter airline/dataset.csv'
df=pd.read_csv(fname,header=0,encoding = 'unicode_escape')
dfpos=df[df.airline_sentiment=="positive"]
in1=dfpos.index
# print(in1)
dfneg=df[df.airline_sentiment!="positive"]
in0=dfneg.index
# print(in0)

vectorizer = CountVectorizer(ngram_range=(1,2))
df['text']=df['text'].apply(clean_text)
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
#     pyplot.boxplot(x,labels=['Not Positive','Positive'])
#     pyplot.grid(True)
#     pyplot.xlabel('Metrics')
#     pyplot.ylabel('95%CI')
#     fna='C:/Users/Shivang/Desktop/Twitter airline/Positive/'+str(i)+".png"
#     pyplot.savefig(fna)
#     pyplot.close()


civ=np.zeros((109553,4))
for i in range(0,109553):
    fv=df2[i]
    civ[i,0:2]=civalue(fv[in0])
    civ[i,2:4]=civalue(fv[in1])

# print(civ.shape)


impft=np.zeros((109553,1))
for i in range(0,109553):
    impft[i,0]=calc(civ[i])

# print(impft.shape)


# Relevant features after confidence interval
relft=[]
for i in range(0,109553):
    if impft[i,0]==1:
        relft.append(i)
# print(len(relft))

df3=pd.DataFrame(x,index=df.index) 
df3.drop(df3.columns[relft],axis=1,inplace=True)
new_ind=range(0,3972)
df3.reindex(columns=new_ind)
# print(df3.shape)


# Remove correlated features out of the remaining features
int1=np.where(impft[:,0]==0)
x1n=x[:,int1[0]]
print(x1n.shape)

pcv=np.zeros((3972,3972))
for i in range(0,3972):
  for j in range(0,3972):
    pcv[i,j]=np.corrcoef(x1n[:,i], x1n[:,j])[0, 1]
    

rem=[]
for i in range(0,3972):
  if i not in rem:
    for j in range(0,3972):
      if j not in rem and j!=i:
        if(abs(pcv[i,j])>=0.7):
          rem.append(j);
print(rem)
print(len(rem))


# Gini values