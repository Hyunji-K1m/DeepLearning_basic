import tqdm
from torch.nn import ModuleList
import tensorflow
import keras
from keras.datasets import mnist
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

learning_rate=1e-3

batch_size=12

train_data=datasets.MNIST(file_path,train=True, download=True, transform=transforms.ToTensor())
test_data=datasets.MNIST(file_path,train=True, download=True, transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size)

class RobustModel(torch.nn.Module):
    def __init__(self):
        super(RobustModel,self).__init__()
        self.keep_prob=0.8
        self.in_dim=28*28*3
        self.out_dim=10
        
        n_channels_1=6
        n_channels_2=16
        
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,n_channels_1,kernel_size=5,stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(n_channels_1,n_channels_2,kernel_size=5,stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc3=torch.nn.Linear(4*4*n_channels_2,120,bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        
        self.layer3=torch.nn.Sequential(
            self.fc3,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )
        self.fc4=torch.nn.Linear(120,80,bias=True)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        
        self.layer4=torch.nn.Sequential(
            self.fc4,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )

        self.fc5=torch.nn.Linear(80,self.out_dim,bias=True)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
    
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)#fc들어가기전 Flatten
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.fc5(out)
        return out
