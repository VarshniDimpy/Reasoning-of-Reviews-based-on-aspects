# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:36:20 2020

@author: Lenovo
"""

import numpy as np
import torch

vocab = np.load('./data/test_vocab.npy')
words_glove = np.load('./data/words_glove_test.npy')
words_domain = np.load('./data/words_domain_test.npy')

not_in_glove = list(set(vocab).difference(set(words_glove)))

not_in_domain = list(set(vocab).difference(set(words_domain)))

weights_matrix_glove = list(np.load('./data/vectors_matrix_glove_test.npy'))
words_glove = list(words_glove)
emb_dim=300
for word in not_in_glove :
    weights_matrix_glove.append(np.random.normal(scale=0.6, size=(emb_dim, )))
    words_glove.append(word)
weights_matrix_glove=np.array(weights_matrix_glove)
words_golve = np.array(words_glove)

weights_matrix_domain = list(np.load('./data/vectots_matrix_domain_test.npy',allow_pickle=True))
words_domain = list(words_domain)
emb_dim=100
for word in not_in_domain :
    weights_matrix_domain.append(np.random.normal(scale=0.6, size=(emb_dim, )))
    words_domain.append(word)
weights_matrix_domain=np.array(weights_matrix_domain)
words_domain = np.array(words_domain)


#weights_matrix_domain=np.load('./data/weights_matrix_domain.npy')
#weights_matrix_glove = np.load('./data/weights_matrix_glove.npy')

weights_glove = []

weights_domain = []

weights_glove.append(np.zeros(300))
weights_domain.append(np.zeros(100))

words_glove=list(words_glove)
words_domain=list(words_domain)
for word in vocab :
    
    index1 = words_glove.index(word)
    index2 = words_domain.index(word)
    
    weights_glove.append(weights_matrix_glove[index1])
    
    weights_domain.append(weights_matrix_domain[index2])

np.save('./data/weights_domain_test.npy',weights_domain)
np.save('./data/weights_glove_test.npy',weights_glove)

  
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


