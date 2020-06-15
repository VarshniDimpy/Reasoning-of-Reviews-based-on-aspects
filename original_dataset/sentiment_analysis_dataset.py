# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:27:29 2020

@author: Lenovo
"""

import transformers
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



class TrainLoader(data.Dataset):

  def __init__(self,sentences,aspect_labels,sentiments,attention_mask):

    self.sentences=sentences
    self.aspect_labels=aspect_labels
    self.sentiments=sentiments
    self.attention_mask=attention_mask
    
    self.length=len(self.sentences)


  def __getitem__(self,idx):
      
      sentence = self.sentences[idx]
      aspect_label = self.aspect_labels[idx]
      sentiment = self.sentiments[idx]
      attention_mask = self.attention_mask[idx]
      
      return sentence,aspect_label,sentiment,attention_mask

  def __len__(self):
      return self.length

            
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.model_class, self.tokenizer_class, self.pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        
        self.drop_out=nn.Dropout(0.1)
        self.conv1=nn.Conv1d(768*2,768,1)
        
        self.output=nn.Linear(768,82)
        
        self.layer1=nn.Linear(82,82)
        self.sentiment_output = nn.Linear(82,82*4)
        
    
    def forward(self,x,aspect_label,attention_masks):
        
        aspect_label.requires_grad=False
        x = self.model(x,attention_masks)
        
        x = torch.cat((x[0][:,-1,:],x[0][:,-2,:]),dim=-1)
        
        x = x.unsqueeze(2)
        
        x=self.drop_out(x)
        
        x=self.conv1(x)
        x=F.relu(x)
        #print(x.shape)
        x=x.squeeze(2)
        x=self.output(x)
        x=F.relu(x)
        x = aspect_label.view(aspect_label.shape[0]*82)*x.view(aspect_label.shape[0]*82)
        
        x= x.view(16,82)
        
        x= self.layer1(x)
        x=F.relu(x)
        
        x = self.sentiment_output(x)
        
        print(x.shape)
        
        x=x.view(aspect_label.shape[0],82,4)
        
        #x=F.softmax(x,dim=2)
        
        return x
    
model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

train_data = np.load('train_data.npy',allow_pickle=True)

sentiment_mapping={}
sentiment_mapping['positive']=1
sentiment_mapping['negative']=2
sentiment_mapping['neutral']=3

all_labels=[]
for row in train_data:
    if row[-1]=='':
        continue
    else:
        all_labels.extend(row[-1]['category'])

all_labels=list(set(all_labels))

labels_mapping={}
idx=0

for label in all_labels:
    labels_mapping[label]=idx
    idx+=1
    
aspect_labels=[]
sentiment=[]
sentences=[]
for row in train_data:
    
    if row[-1]=='':
        continue
    else:
        labels=row[-1]['category']
        sentiments=row[-1]['polarity']
        a=[]
        s=[]
        for i in labels:
            a.append(labels_mapping[i])
        for j in sentiments:
            s.append(sentiment_mapping[j])
        label = a
    sentence=row[2]
    sentences.append(sentence)
    aspect_labels.append(label)
    sentiment.append(s)

print(len(sentences),len(aspect_labels),len(sentiment))

classes = len(all_labels)+1

new_aspect_labels=[]
new_sentiments =[]
#https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203/4

for label,sen in zip(aspect_labels,sentiment):
    a=torch.LongTensor(label)
    a=a.unsqueeze(0)
    new = np.zeros((classes,1))
    
    
    for index,idc in zip(label,sen):
        new[index][0]=idc
    new_sentiments.append(new)
    target = torch.zeros(a.size(0),82).scatter_(1, a, 1.)
    new_aspect_labels.append(target.numpy()[0])

new_aspect_labels=torch.FloatTensor(np.array(new_aspect_labels))

new_sentiments=torch.FloatTensor(np.array(new_sentiments))

tokenized_sentences = []

for s in sentences :
    if isinstance(s,float):
        print("hi there!")
        continue
    a=tokenizer.encode(s,add_sepcial_tokens=True)
    tokenized_sentences.append(a)

max_len = 0
for i in tokenized_sentences:
    if len(i) > max_len:
        max_len = len(i)


padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_sentences])
input_ids = torch.LongTensor(np.array(padded))
attention_mask = np.where(padded != 0, 1, 0)

train_target=TrainLoader(input_ids,new_aspect_labels,new_sentiments,attention_mask)

train_loader=torch.utils.data.DataLoader(train_target,batch_size=16,shuffle=True)

criterion = nn.CrossEntropyLoss()
#train
net=Model()

optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

for epoch in range(10):
    total_loss=0
    for sentence,aspect_label,sentiment,mask in train_loader:
        
        pred=net(sentence,aspect_label,mask)
        
        loss = criterion(pred[0],aspect_label[0].long())
        
        optimizer.zero_grad()
        
        for i in range(1,len(pred)):
            loss += criterion(pred[i],aspect_label[i].long())
        
        loss.backward()
        
        optimizer.step()
        total_loss+=loss.item()
        print(loss.item())
    print("epoch : "epoch," loss : ",loss.item())

        