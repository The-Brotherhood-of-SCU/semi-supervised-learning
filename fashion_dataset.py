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
    if isDiv255:
        tensor=tensor/255.0

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

train_dataset=TrainDataSet()
unlabeled_dataset=UnlabeledDataSet()
test_dataset=TestDataSet()
final_dataset=FinalTestDataSet()
