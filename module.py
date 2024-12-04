import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov1=nn.Conv2d(1,10,3,padding=1)
        self.cov2=nn.Conv2d(10,2,3,padding=1)
        self.maxpool=nn.MaxPool2d(2,2)
        self.linear=nn.Linear(2*7*7,10)
    def forward(self,x:torch.Tensor):
        # input shape is batchx784
        x=x.view(-1,1,28,28)
        x=self.cov1(x)
        x=self.maxpool(x)
        x=F.relu(x)
        # shape is batchx10x14x14
        x=self.cov2(x)
        x=self.maxpool(x)
        x=F.relu(x)
        # shape is batchx2x7x7
        x=x.view(-1,2*7*7)
        x=self.linear(x)
        x=F.softmax(x,dim=1)
        return x