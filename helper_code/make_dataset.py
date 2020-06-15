# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:42:33 2020

@author: Lenovo
"""

import numpy as np
import string
from textblob import TextBlob

train_data=np.load('test_data.npy',allow_pickle=True)
vocab = np.load('./data/test_vocab.npy')
all_labels=[]

for row in train_data:
    if row[-1]=='':
        continue
    else:
        all_labels.extend(row[-1]['category'])

all_labels=list(set(all_labels))

print("number of classes = ",len(all_labels))

labels_mapping={}

idx=0
labels_mapping['NIL']=0
for label in all_labels:
    idx+=1
    labels_mapping[label]=idx
        
x_train=[]
y_train=[]
vocab = list(vocab)
for row in train_data:
    sentence = row[2]
    words = sentence.split()
    table = str.maketrans('','',string.punctuation)
    stripped_words = [w.translate(table) for w in words]
    new_words=[str(TextBlob(w).correct()) for w in stripped_words]
    final_words=[w.lower() for w in new_words]
    
    train=[]
    for word in final_words:
        index=vocab.index(word)
        train.append(index+1)
    x_train.append(train)
    
    if row[-1]=='':
        y_train.append([0])
        continue
    a=[]
    labels = row[-1]['category']
    for i in labels:
        
        a.append(labels_mapping[i])
    y_train.append(a)


