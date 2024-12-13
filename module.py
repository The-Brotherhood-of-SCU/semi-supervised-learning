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

    def forward(self, x: torch.Tensor):
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
        
        x = x.flatten(1)
        x = torch.cat((x, x2), dim=1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x