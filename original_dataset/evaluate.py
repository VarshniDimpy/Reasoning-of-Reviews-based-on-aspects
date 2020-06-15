# -*- coding: utf-8 -*-
"""
Created on Fri May 15 05:34:01 2020

@author: Lenovo
"""
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

test_loader=torch.utils.data.DataLoader(train_target,batch_size=16,shuffle=True)

predictions=[]
gt=[]

for sentences,labels,mask in train_loader:
    sentences=sentences.cuda()
    
    preds=net(sentences,mask)
    #print(preds.shape)
    preds=F.sigmoid(preds)
    #print(preds.shape)
    
    labels_=labels.cpu().detach().numpy().flatten()
    gt.extend(labels_)
    
    preds=preds.cpu().detach().numpy().flatten()
    
    for p in preds:
        
        if p>0.2:
            predictions.append(1)
        else:
            predictions.append(0)

gt=np.array(gt)
predictions = np.array(predictions)
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

c=classification_report(gt,predictions)

print(c.split('\n'))
