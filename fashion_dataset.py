import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np





device="cuda:0" if torch.cuda.is_available() else "cpu"
floder_name="fashion-dataset"
print("current_device: ",device)
def load_npy(file_name:str,isDiv255=True)->torch.Tensor:
    tensor=torch.tensor(np.load(f"{floder_name}/{file_name}")).to(device).permute(1,0)
    tensor=tensor.float() 
    if isDiv255:#feature
        tensor=tensor/255.0
    else:#label
        tensor=tensor.argmax(1)
    return tensor
class TrainDataSet(Dataset):
    def __init__(self):
        self.x=load_npy("train_x.npy")
        self.y=load_npy("train_y.npy",isDiv255=False)
        print("x-shape",self.x.shape,self.x.dtype)
        print("y-shape",self.y.shape)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return len(self.x)

class EnhancedDataset(Dataset):
    def __init__(self, original_dataset,offset=2):
        self.x = []
        self.y = []

        for data, label in original_dataset:
            # 将数据重塑为28x28的图像
            image = data.view(28, 28)
            
            # 左右上下移动3个像素
            left = torch.roll(image, shifts=-offset, dims=1)
            right = torch.roll(image, shifts=offset, dims=1)
            up = torch.roll(image, shifts=-offset, dims=0)
            down = torch.roll(image, shifts=offset, dims=0)

            # 将修改后的图像添加到数据列表中，并保持标签不变
            self.x.append(left.view(-1))
            self.x.append(right.view(-1))
            self.x.append(up.view(-1))
            self.x.append(down.view(-1))
            self.y.extend([label] * 4)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class UnlabeledDataSet(Dataset):
    def __init__(self):
        self.x=load_npy("unlabeled_x.npy")
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)

class TestDataSet(Dataset):
    def __init__(self):
        self.x=load_npy("test_x.npy")
        self.y=load_npy("test_y.npy",isDiv255=False)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return len(self.x)

class FinalTestDataSet(Dataset):
    def __init__(self):
        self.x=load_npy("final_x.npy")
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)
    
class RotatedDataset(Dataset):
    def __init__(self, original_dataset):
        self.x = []
        self.y = []

        for data, label in original_dataset:
            # 将数据重塑为28x28的图像
            image = data.view(28, 28)
            # 保存原始图像
            self.x.append(image)
            self.y.append(label)
            # 旋转90度
            rotated90 = torch.rot90(image, k=1, dims=(0, 1))
            self.x.append(rotated90)
            self.y.append(label)
            # 旋转180度
            rotated180 = torch.rot90(image, k=2, dims=(0, 1))
            self.x.append(rotated180)
            self.y.append(label)
            # 旋转270度
            rotated270 = torch.rot90(image, k=3, dims=(0, 1))
            self.x.append(rotated270)
            self.y.append(label)

        # 将图像重塑回784维向量
        self.x = [img.reshape(-1) for img in self.x]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class FlippedDataset(Dataset):
    def __init__(self, original_dataset):
        self.x = []
        self.y = []

        for data, label in original_dataset:
            # 将数据重塑为28x28的图像
            image = data.view(28, 28)
            # 保存原始图像
            self.x.append(image)
            self.y.append(label)
            # 水平翻转
            flipped = torch.flip(image, dims=(1,))
            self.x.append(flipped)
            self.y.append(label)

        # 将图像重塑回784维向量
        self.x = [img.reshape(-1) for img in self.x]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class CombinedUnlabeledDataset(Dataset):
    def __init__(self, unlabeled_dataset, offset=2):
        self.x = []

        for data in unlabeled_dataset: #6x数据集增加
            # 将数据重塑为28x28的图像
            image = data.view(28, 28)
            # 保存原始图像
            self.x.append(image)
            # # 旋转90度
            # rotated90 = torch.rot90(image, k=1, dims=(0, 1))
            # self.x.append(rotated90)
            # # 旋转180度
            # rotated180 = torch.rot90(image, k=2, dims=(0, 1))
            # self.x.append(rotated180)
            # # 旋转270度
            # rotated270 = torch.rot90(image, k=3, dims=(0, 1))
            # self.x.append(rotated270)
            # 水平翻转
            flipped = torch.flip(image, dims=(1,))
            self.x.append(flipped)
            # 左右上下移动offset个像素
            left = torch.roll(image, shifts=-offset, dims=1)
            right = torch.roll(image, shifts=offset, dims=1)
            up = torch.roll(image, shifts=-offset, dims=0)
            down = torch.roll(image, shifts=offset, dims=0)
            self.x.append(left.view(-1))
            self.x.append(right.view(-1))
            self.x.append(up.view(-1))
            self.x.append(down.view(-1))

        # 将图像重塑回784维向量
        self.x = [img.reshape(-1) for img in self.x]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

train_dataset=TrainDataSet()
unlabeled_dataset=UnlabeledDataSet()
test_dataset=TestDataSet()
final_dataset=FinalTestDataSet()
# 直接用这个替代原始的
train_dataset=FlippedDataset(train_dataset)
combined_unlabeled_dataset = CombinedUnlabeledDataset(unlabeled_dataset)

enhanced_dataset_1=EnhancedDataset(train_dataset,offset=1)

batch_size=32
# loaders
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
unlabeled_loader=DataLoader(combined_unlabeled_dataset,batch_size=batch_size,shuffle=True) 
test_loader=DataLoader(test_dataset,batch_size=256,shuffle=True)
final_test_loader=DataLoader(final_dataset,batch_size=batch_size,shuffle=True)
enhance_loader_1=DataLoader(enhanced_dataset_1,batch_size=batch_size,shuffle=True)


def transform_offset(x:torch.Tensor,offset=1):
    x=x.view(-1,28,28)
    left = torch.roll(x, shifts=-offset, dims=2)
    right = torch.roll(x, shifts=offset, dims=2)
    up = torch.roll(x, shifts=-offset, dims=1)
    down = torch.roll(x, shifts=offset, dims=1)
    return [left,right,up,down]
