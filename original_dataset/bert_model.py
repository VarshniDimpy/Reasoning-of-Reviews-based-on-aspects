
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

model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
all_labels=[]
train_data=np.load('train_data.npy',allow_pickle=True)
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
for row in train_data:
    sentence = row[2]
    
    x_train.append(sentence)
    
    if row[-1]=='':
        y_train.append([0])
        continue
    a=[]
    labels = row[-1]['category']
    for i in labels:
        
        a.append(labels_mapping[i])
    y_train.append(a)

tokenized_sentences = []
labels=[]

for (s,l) in zip(x_train,y_train) :
    if isinstance(s,float):
        continue
    a=tokenizer.encode(s,add_sepcial_tokens=True)
    tokenized_sentences.append(a)
    labels.append(l)

max_len = 0
for i in tokenized_sentences:
    if len(i) > max_len:
        max_len = len(i)


padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_sentences])
input_ids = torch.LongTensor(np.array(padded))
attention_mask = np.where(padded != 0, 1, 0)

'''
with torch.no_grad():
    last_hidden_states = model(input_ids)
'''

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.model_class, self.tokenizer_class, self.pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        
        self.drop_out=nn.Dropout(0.1)
        self.conv1=nn.Conv1d(768*2,768,1)
        
        self.output=nn.Linear(768,82)
    
    def forward(self,x,attention_masks):
        
        x = self.model(x,attention_masks)
        
        x = torch.cat((x[0][:,-1,:],x[0][:,-2,:]),dim=-1)
        
        x = x.unsqueeze(2)
        
        x=self.drop_out(x)
        
        x=self.conv1(x)
        x=F.relu(x)
        #print(x.shape)
        
        x=x.squeeze(2)
        x=self.output(x)
        
        return x
    
class TargetLoader(data.Dataset):

  def __init__(self,sentences,new_y_train,attention_mask):

    self.sentences=sentences
    self.new_y_train=new_y_train
    self.attention_mask=attention_mask
    
    self.length=len(self.sentences)


  def __getitem__(self,idx):

    image=self.sentences[idx]
    label=self.new_y_train[idx]
    mask = self.attention_mask[idx]
    return image,label,mask
  
  def __len__(self):

    return self.length


new_y_train=[]
#https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203/4

for label in y_train:
    a=torch.LongTensor(label)
    a=a.unsqueeze(0)
    target = torch.zeros(a.size(0),82).scatter_(1, a, 1.)
    new_y_train.append(target.numpy()[0])

new_y_train=torch.FloatTensor(np.array(new_y_train))


train_target=TargetLoader(input_ids,new_y_train,attention_mask)

train_loader=torch.utils.data.DataLoader(train_target,batch_size=16,shuffle=True)

net = Model()
#net = net.cuda()
optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)
criterion=nn.BCEWithLogitsLoss()

for epoch in range(5):
    
    total_loss=0
    for sentences,labels,mask in train_loader:
        #break
        #sentences = sentences.cuda()
        #labels = labels.cuda()
        #mask=mask.cuda()
        preds = net(sentences,mask)
        
        loss = criterion(preds,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss+=loss.item()
        
        print("loss :",loss.item())
    print("epoch done !",epoch, " loss : ",total_loss)