import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(4)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(64 * 7 * 7 + 4 * 14 * 14, 128)
        self.linear2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.2)
    def forward(self, x: torch.Tensor):
        # input shape is batchx784
        x = x.view(-1, 1, 28, 28)
        x2 = self.conv12(x)
        x2 = self.bn12(x2)
        x2 = F.relu(x2)
        x2 = self.maxpool(x2)  # x2 shape is batchx4x14x14
        x2=self.dropout(x2)
        x2 = x2.flatten(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x=self.dropout(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = x.flatten(1)
        x = torch.cat((x, x2), dim=1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
# 这个是拿来尝试的，效果不咋样，不用写在报告里面
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(4)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(64 * 7 * 7 + 4 * 14 * 14, 128)
        self.linear2 = nn.Linear(128, 10)
        self.blend_x_pre=nn.Linear(256,128)
        self.blend2=nn.Linear(128,128)

        self.pre2high = nn.Linear(10, 128)

    def forward(self, x: torch.Tensor,pre:torch.Tensor):
        # input shape is batchx784
        x = x.view(-1, 1, 28, 28)
        x2 = self.conv12(x)
        x2 = self.bn12(x2)
        x2 = F.relu(x2)
        x2 = self.maxpool(x2)  # x2 shape is batchx4x14x14
        x2 = x2.flatten(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        pre=self.pre2high(pre)
        pre=F.relu(pre)
        
        x = x.flatten(1)
        x = torch.cat((x, x2), dim=1)
        x = self.linear(x)

        x=self.blend_x_pre(torch.cat((x,pre),dim=1))
        x=F.relu(x)
        x=self.blend2(x)
        x=F.relu(x)

        x = F.relu(x)
        x = self.linear2(x)
        return x



class Net3(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(4)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(64 * 7 * 7 + 4 * 14 * 14, 128)
        self.linear2 = nn.Linear(128, 10)
        self.encoded_linear=nn.Linear(64 * 7 * 7 + 4 * 14 * 14,64*7*7+4*14*14)
        self.decoder = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 4 * 14 * 14, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, output_padding=1,padding=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.1)

    def encode(self, x: torch.Tensor):
        x = x.view(-1, 1, 28, 28)
        x2 = self.conv12(x)
        x2 = self.bn12(x2)
        x2 = F.relu(x2)
        x2 = self.maxpool(x2)  # x2 shape is batchx4x14x14
        x2 = self.dropout(x2)
        x2 = x2.flatten(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        encoded = x.flatten(1)
        encoded= torch.cat((encoded, x2),dim=1)
        encoded=self.encoded_linear(encoded)
        encoded=F.relu(encoded)
        return encoded

    def decode(self, encoded: torch.Tensor):
        decoded = self.decoder(encoded)
        
        return decoded.flatten(1)

    def classify(self, encoded: torch.Tensor):
        x = self.linear(encoded)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        classification = self.classify(encoded)
        return classification, decoded