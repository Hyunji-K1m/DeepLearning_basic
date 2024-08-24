#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
import tensorflow as tf
from module import CNN


# In[2]:


import numpy as np
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
import tensorflow
import keras
from keras.datasets import mnist


# In[3]:


device='cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device== 'cuda':
    torch.cuda.manual_seed_all(777)


# In[4]:


batch_size=12

train_data=datasets.MNIST('C:\\Users\\82104\\Downloads',train=True, download=True, transform=transforms.ToTensor())
test_data=datasets.MNIST('C:\\Users\\82104\\Downloads',train=True, download=True, transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size)


# In[5]:


def test(data_loader,model):
    model.eval()
    n_predict=0
    n_correct=0
    with torch.no_grad():
        for X,Y in tqdm.tqdm(data_loader,desc='data_loader'):
            y_hat=model(X)
            y_hat.argmax()
            
            _,predicted=torch.max(y_hat,1)
            
            n_predict += len(predicted)
            n_correct += (Y==predicted).sum()
            
    accuracy=int(n_correct)/n_predict 
    print(f"Accuracy:{accuracy}()")


# In[6]:


model=CNN()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch:0.95** epoch, last_epoch=-1)


# In[10]:


training_epochs=5

for epoch in range(training_epochs):
    model.train()
    cost=0
    n_batches=0
    for X,Y in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        y_hat=model(X)
        loss=criterion(y_hat,Y)
        loss.backward()
        optimizer.step()
        
        cost+=loss.item()
        n_batches+=1
    
    cost/=n_batches
    print('[Epoch: {:>4}] cost = {:>9}'.format(epoch + 1, cost))
    print('Dev')
    test(test_loader,model)


# In[11]:


torch.save(model.state_dict(), 'model.pt')


# In[ ]:




