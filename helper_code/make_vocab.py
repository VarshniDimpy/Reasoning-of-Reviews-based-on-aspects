# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:37:10 2020

@author: Lenovo
"""

import numpy as np
import string
from textblob import TextBlob

train_data=np.load('train_data.npy',allow_pickle=True)

test_data=np.load('test_data.npy',allow_pickle=True)

vocab =[]
i=0
for row in train_data:
    i+=1
    print(i)
    sentence = row[2]
    words = sentence.split()
    table = str.maketrans('','',string.punctuation)
    stripped_words = [w.translate(table) for w in words]
    new_words=[str(TextBlob(w).correct()).lower() for w in stripped_words]
    vocab.extend(new_words)

vocab = list(set(vocab))
vocab = [w.lower() for w in vocab]

vocab = list(set(vocab))
print("total # of train vocab : ",len(vocab))
np.save('./data/vocab_new.npy',vocab)

i=0
test_vocab=[]
for row in test_data:
    sentence = row[2]
    i+=1
    print(i)
    words = sentence.split()
    table = str.maketrans('','',string.punctuation)
    stripped_words = [w.translate(table) for w in words]
    new_words=[str(TextBlob(w).correct()).lower() for w in stripped_words]
    
    test_vocab.extend(new_words)

test_vocab = list(set(test_vocab))

test_vocab = [w.lower() for w in test_vocab]

test_vocab = list(set(test_vocab))
np.save('./data/test_vocab.npy',test_vocab)

print("total # of train vocab : ",len(vocab))
np.save('./data/vocab_new.npy',vocab)
print("total # of train vocab : ",len(list(set(test_vocab))))
    
np.save('./data/both_vocab.npy',list(set(test_vocab)))    