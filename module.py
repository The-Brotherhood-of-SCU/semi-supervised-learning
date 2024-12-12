import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov1=nn.Conv2d(1,32,3,padding=1)
        self.cov2=nn.Conv2d(32,16,3,padding=1)
        self.cov12=nn.Conv2d(1,4,3,padding=1)
        self.maxpool=nn.MaxPool2d(2,2)
        self.linear=nn.Linear(16*7*7+4*14*14,64)
        self.linear2=nn.Linear(64,10)
    def forward(self,x:torch.Tensor):
        # input shape is batchx784
        x=x.view(-1,1,28,28)
        x2=self.cov12(x)
        x2=self.maxpool(x2)#x2 shape is batchx4x14x14
        x2=x2.flatten(1)
        x=self.cov1(x)
        x=self.maxpool(x)
        x=F.relu(x)
        # shape is batchx32x14x14
        x=self.cov2(x)
        x=self.maxpool(x)
        x=F.relu(x)
        # shape is batchx64x7x7
        x=x.flatten(1)
        x=torch.cat((x,x2),dim=1)
        x=self.linear(x)
        x=F.relu(x)
        x=self.linear2(x)
        #x=F.softmax(x,dim=1)
        return x