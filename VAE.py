from fashion_dataset import *
from module import Net4
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# Initialize the model
model = Net4().to(device)

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Define loss function and optimizer
criterion_reconstruct = torch.nn.MSELoss(reduction='sum')  # Use MSELoss for reconstruction
criterion_class = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-5)
def get_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 更稳定的KL散度计算
    logvar = logvar.clamp(min=-20, max=20)  # 限制logvar范围
    return 0.5 * torch.sum(
        torch.exp(logvar) + mu.pow(2) - 1 - logvar, 
        dim=1
    ).mean()

# Training loop
def train_semi_supervised(model=model, train_loader=enhance_loader_1, unlabeled_loader=unlabeled_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_corrects = 0
        for ((labeled_data, labels), unlabeled_data) in zip(train_loader, unlabeled_loader):
            # Forward pass
            optimizer.zero_grad()
            # 修改优化器初始化，添加权重衰减
            
            
            # 修改训练循环中的损失计算
            labeled_class, decoded, mu, logvar = model(labeled_data)
            reconstruction_loss = criterion_reconstruct(decoded, labeled_data) * 0.1
            class_loss = criterion_class(labeled_class, labels) * 20
            kl_loss = get_kl_loss(mu, logvar) * 0.05
            total_loss=reconstruction_loss+kl_loss*0.2+class_loss
            preds = labeled_class.argmax(1).detach()
            train_corrects += (preds==labels.data).sum()
            # Backward pass
            total_loss.backward()
            # avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # unlabeled
            optimizer.zero_grad() 
            _,decoded,mu,logvar = model(unlabeled_data)
            reconstruction_loss=criterion_reconstruct(decoded,unlabeled_data)
            kl_loss=get_kl_loss(mu,logvar)
            total_loss=reconstruction_loss+kl_loss*0.05
            # Backward pass
            total_loss.backward()
            optimizer.step()
        acc=(train_corrects / len(train_loader.dataset)).item()
        print(f'Epoch {epoch+1}/{epochs},total Loss: {total_loss.item():.4f},construction Loss: {reconstruction_loss.item():.4f},class acc: {acc:.4f}')
def test(test_loader=test_loader, net=model,isOffset=True):
    # set model to eval mode
    net.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = F.softmax(net(inputs)[0],dim=-1)
            if isOffset:
                for i in transform_offset(inputs):
                    outputs+=F.softmax(net(i)[0],dim=-1)
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
    getacc=(corrects / len(test_loader.dataset)).item()
    print("get new accuracy::{}".format(getacc))
    return getacc

highest_acc=0
highest_acc_data=None
def update_final_output():
    global highest_acc,highest_acc_data
    model.eval()
    test_acc=test()
    if test_acc>highest_acc:
        print("new higher test_acc",test_acc)
    else:
        print("test_acc not higher",test_acc,"highest_acc",highest_acc)
        return
    with torch.no_grad():
        x=torch.stack([final_dataset[i] for i in range(len(final_dataset))])
        y=F.softmax(model(x)[0],dim=-1)
        for i in transform_offset(x):
            y+=F.softmax(model(i)[0],dim=-1)
        y=y.argmax(1).detach().cpu().unsqueeze(0)
        y=y.numpy()
    highest_acc=test_acc
    highest_acc_data=y
    
def draw(index:int=0):
    import matplotlib.pyplot as plt
    with torch.no_grad():
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        data:torch.Tensor=unlabeled_dataset[index]
        _,data2,_,_=model(data)
        data2=data2.view(28,28)
        # 显示原始图像
        ax[0].imshow(data.cpu().view(28,28),cmap="gray")
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        # 显示掩码
        ax[1].imshow(data2.cpu().view(28,28), cmap='gray')
        ax[1].set_title('reconstructed Image')
        ax[1].axis('off')
def save_final():
    import os
    os.makedirs("out3",exist_ok=True)
    print("shape",highest_acc_data.shape)
    np.save(f"out3/output_{highest_acc}.npy",highest_acc_data)

import random
loaders = [enhance_loader_1, enhance_loader_2, train_loader, train_loader]

def train_infty():
    while True:
        loader = random.choice(loaders)
        train_semi_supervised(train_loader=loader, epochs=1)
        update_final_output()

if __name__ == "__main__":
    train_infty()
