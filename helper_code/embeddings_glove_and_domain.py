# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:43:44 2020

@author: Lenovo
"""

#https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

with open('../laptop_emb.vec') as f:
   lines_domain = f.readlines()
   
with open('../glove.840B.300d.txt',encoding="utf-8") as f:
   lines_glove = f.readlines()


#What we need to do at this point is to create an embedding layer, 
#that is a dictionary mapping integer indices (that represent words)
# to dense vectors.

import numpy as np

vocab = np.load('./data/test_vocab.npy',allow_pickle=True)
vocab=list(vocab)

words_glove = []
idx_glove=0
word2idx_glove = {}

vectors_matrix_glove=[]

for line in lines_glove :
    l=line.split(" ")
    word = l[0]
    if word in vocab:
        words_glove.append(word)
        word2idx_glove[word] = idx_glove
        em = np.array(l[1:]).astype(np.float)
        vectors_matrix_glove.append(em)
        idx_glove+=1

np.save('./data/vectors_matrix_glove_test.npy',vectors_matrix_glove)
np.save('./data/words_glove_test.npy',words_glove)

vectors_matrix_glove = np.array(vectors_matrix_glove)
print(vectors_matrix_glove.shape)
words_domain = []
idx_domain=0
word2idx_domain = {}

vectors_matrix_domain=[]

for line in lines_domain :
    l=line.split()
    word = l[0]
    if word in vocab:
        words_domain.append(word)
        word2idx_domain[word] = idx_domain
        em = np.array(l[1:]).astype(np.float)
        vectors_matrix_domain.append(em)
        idx_domain+=1

vectors_matrix_domain = np.array(vectors_matrix_domain)
print(vectors_matrix_domain.shape)    
np.save('./data/vectots_matrix_domain_test.npy',vectors_matrix_domain)
np.save('./data/words_domain_test.npy',words_domain)