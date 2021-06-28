
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import torch
import transformers as ppb 


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


from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTModel.from_pretrained('openai-gpt')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]



def gpt_fscore(text,model,tokenizer):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0) 
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return last_hidden_states


fname='C:/Users/Shivang/Desktop/Twitter airline/dataset.csv'
df = pd.read_csv(fname,encoding='latin-1')
df['text'] = df['text'].apply(clean_text)
print(df.head(10))
train_tokens = list(map(lambda t:   tokenizer.tokenize(t)+['[CLS]'], df['text']))
train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids,train_tokens))
b=np.zeros((np.shape(df)[0]))
for i in range(0,np.shape(df)[0]):
    b[i]=np.shape(train_tokens_ids[i])[0]
pad=np.zeros((np.shape(df)[0],int(np.max(b))))
for i in range(0,np.shape(df)[0]):
    a=train_tokens_ids[i]
    for j in range(0,np.shape(a)[0]):
        pad[i,j]=a[j]
        
print(pad.shape)

pad1=pad[12000:,:]  
input_ids = torch.tensor(np.array(pad1))        
with torch.no_grad():
    last_hidden_states = model(input_ids.long())    
features = last_hidden_states[0][:,0,:].numpy()
fname='C:/Users/Shivang/Desktop/Twitter airline/gptmodel(12000-end).csv'
np.savetxt(fname,features, delimiter=',', fmt='%f')     
