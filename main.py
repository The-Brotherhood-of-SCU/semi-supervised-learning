import torch
from fashion_dataset import *
from module import *
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch import optim




batch_size=32
epochs=30

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
def _train_supervised(train_loader, net, loss_fn=loss_fn):
    # set model to train mode
    net.train()
    for i in range(epochs):
        corrects = 0
        optimizer=get_optimizer(net)
        #for inputs, labels in tqdm(train_loader, leave=False):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs:torch.Tensor = net(inputs)
            loss:torch.Tensor = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
            acc=(corrects / len(train_loader.dataset)).item()
        print("epoch",i,"loss",loss.item(),"acc",acc)
    return loss, acc



def _train_semi_supervised(train_loader, net, loss_fn=loss_fn):
    # set model to train mode
    net.train()
    for i in range(epochs):
        corrects = 0
        optimizer=get_optimizer(net)
        #for inputs, labels in tqdm(train_loader, leave=False):
        for inputs in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                labels=net(inputs).argmax(1).detach()
            outputs:torch.Tensor = net(inputs)
            loss:torch.Tensor = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
        print("epoch",i,"loss",loss.item())
    return loss, (corrects / len(train_loader.dataset)).item()
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



def train_supervised():
    train_data=_train_supervised(train_loader,net)
    print("train accuarcy: ",train_data[1])
def train_semi_supervised():
    train_data=_train_semi_supervised(unlabeled_loader,net)
    print("train accuarcy: ",train_data[1])
def test():
    test_data=_test(test_loader,net)
    print("test: ",test_data)

print("start supervised")
train_supervised()
print("start test")
test()
print("start semi supervised")
#train_semi_supervised()
print("start test")
test()