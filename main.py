import torch
from fashion_dataset import *
from module import *
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch import optim




batch_size=32
epochs=10

# loaders
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
unlabeled_loader=DataLoader(unlabeled_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
final_test_loader=DataLoader(final_dataset,batch_size=batch_size,shuffle=True)
#net
net=Net().to(device)

def display_photo(i:int=0):
    import numpy as np
    #display the photo
    import matplotlib.pyplot as plt
    # 显示张量图片
    data:torch.Tensor=train_dataset[i][0]
    data=data.view(28,28)
    plt.imshow(data.cpu(), cmap='gray')
    plt.colorbar()
    plt.show()

def get_optimizer(net):
    return optim.Adam(net.parameters())
loss_fn = nn.CrossEntropyLoss()
def _train_supervised( train_loader_,net,loss_fn=loss_fn,epochs=epochs):
    # set model to train mode
    net.train()
    for i in range(epochs):
        corrects = 0
        optimizer=get_optimizer(net)
        #for inputs, labels in tqdm(train_loader, leave=False):
        for inputs, labels in train_loader_:
            optimizer.zero_grad()
            outputs:torch.Tensor = net(inputs)
            loss:torch.Tensor = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
            acc=(corrects / len(train_loader_.dataset)).item()
        print("epoch",i,"loss",loss.item(),"acc",acc)
    return loss, acc



def semi_supervised_training_with_regularization(unlabeled_dataloader, labeled_dataloader, model, criterion=loss_fn, optimizer_getter=get_optimizer, num_epochs=epochs, lambda_l2=0.01):
    """
    进行带L2正则化的半监督学习的训练过程。

    参数:
    - unlabeled_dataloader: 无标签数据集的DataLoader
    - labeled_dataloader: 有标签数据集的DataLoader
    - model: 要训练的模型
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练的轮数
    - lambda_l2: L2正则化的系数
    """
    model.train()  # 设置模型为训练模式
    optimizer=optimizer_getter(model)
    for epoch in range(num_epochs):
        # 为无标签的数据生成伪标签并进行训练
        for data,(x,y) in zip(unlabeled_dataloader,labeled_dataloader):
            optimizer.zero_grad()
            outputs:torch.Tensor = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            with torch.no_grad():  # 不计算生成伪标签的梯度
                pseudo_labels = model(data).argmax(dim=1)  # 生成伪标签
            outputs = model(data)
            loss = criterion(outputs, pseudo_labels)  # 使用伪标签计算损失
            
            # 添加L2正则化
            l2_reg = torch.tensor(0.).to(data.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += lambda_l2 * l2_reg
            
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}] completed')


def _test(test_loader, net):
    # set model to eval mode
    net.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, leave=False):
            outputs = net(inputs)
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
    return (corrects / len(test_loader.dataset)).item()



def train_supervised(epochs=epochs):
    print("start supervised")
    train_data=_train_supervised(train_loader,net,epochs=epochs)
    print("train accuarcy: ",train_data[1])
def train_supervised_enhanced(offset=2):
    enhanced_dataset=EnhancedDataset(train_dataset,offset=offset)
    train_loader_=DataLoader(enhanced_dataset,batch_size=batch_size,shuffle=True)
    print("start supervised_enhanced with offset: ",offset)
    train_data=_train_supervised(train_loader_,net)
    print("train accuarcy: ",train_data[1])
def train_supervised_rotated():
    print("start supervised_rotated")
    enhanced_dataset=RotatedDataset(train_dataset)
    train_loader_=DataLoader(enhanced_dataset,batch_size=batch_size,shuffle=True)
    train_data=_train_supervised(train_loader_,net)
    print("train accuarcy: ",train_data[1])
def train_semi_supervised(lambda_l2=0.001):
    print("start semi supervised")
    train_data=semi_supervised_training_with_regularization(unlabeled_loader,train_loader,net,lambda_l2=lambda_l2)
    #print("train accuarcy: ",train_data[1])
def test():
    print("start test")
    test_data=_test(test_loader,net)
    print("test: ",test_data)

# --- TRAIN ---
train_supervised_rotated()
test()

train_semi_supervised()
test()

train_supervised_enhanced(3)
test()

train_semi_supervised(lambda_l2=0.1)
test()

train_supervised_enhanced(1)
test()

train_semi_supervised()
test()





