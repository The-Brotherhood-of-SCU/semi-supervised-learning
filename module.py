import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

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


#simple encode-decode
class Net3(nn.Module): 
    def __init__(self):
        super().__init__()
        hidden_size = 80
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(4)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, 10)
        self.encoded_linear = nn.Linear(64 * 7 * 7 + 4 * 14 * 14, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 32 * 7 * 7),
            nn.ReLU(),
            nn.Linear(32 * 7 * 7, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 48, 3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 1, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.2)

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
        encoded = torch.cat((encoded, x2), dim=1)
        encoded = self.encoded_linear(encoded)
        encoded = F.relu(encoded)
        return encoded

    def decode(self, encoded: torch.Tensor):
        decoded = self.decoder(encoded)
        return decoded.flatten(1)

    def classify(self, encoded: torch.Tensor):
        x = self.dropout(encoded)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        classification = self.classify(encoded)
        return classification, decoded


class Net4(nn.Module): 
    def __init__(self):
        super().__init__()
        hidden_size = 80
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(4)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, 10)
        self.encode_linear1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.encode_linear2 = nn.Linear(4 * 14 * 14, hidden_size)
        self.std_fc = nn.Linear(hidden_size*2, hidden_size)  # 添加这行
        self.logvar_fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 解码器添加skip connection
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 32 * 7 * 7),
            nn.ReLU(),
            nn.Linear(32 * 7 * 7, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 48, 3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # 在最后一层前添加
            nn.ConvTranspose2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 1, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.2)

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
        
        # 修改编码器线性层
        x1 = self.encode_linear1(x.flatten(1))
        x2 = self.encode_linear2(x2.flatten(1))
        x = torch.cat((x1, x2), dim=1)
        
        mu = self.std_fc(x)
        logvar = self.logvar_fc(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        # 添加更严格的数值检查
        logvar = logvar.clamp(min=-20, max=20)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, encoded: torch.Tensor):
        decoded = self.decoder(encoded)
        return decoded.flatten(1)

    def classify(self, encoded: torch.Tensor):
        x = self.dropout(encoded)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def forward(self, x: torch.Tensor):
        mu,logvar = self.encode(x)
        sample=self.reparameterize(mu,logvar)
        decoded = self.decode(sample)
        classification = self.classify(sample)
        return classification, decoded,mu,logvar
