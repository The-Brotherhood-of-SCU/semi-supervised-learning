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

train_dataset=TrainDataSet()
unlabeled_dataset=UnlabeledDataSet()
test_dataset=TestDataSet()
final_dataset=FinalTestDataSet()
