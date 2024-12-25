from fashion_dataset import *
from module import Net3
import torch
import torch.optim as optim
import torch.nn.functional as F

# Initialize the model
model = Net3().to(device)

# Define loss function and optimizer
criterion_reconstruct = torch.nn.MSELoss()  # Use MSELoss for reconstruction
criterion_class = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters())

# Training loop
def train_semi_supervised(model=model, train_loader=enhance_loader_1, unlabeled_loader=unlabeled_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_corrects = 0
        for ((labeled_data, labels), unlabeled_data) in zip(train_loader, unlabeled_loader):
            # Forward pass
            optimizer.zero_grad()
            labeled_class,decoded = model(labeled_data)
            reconstruction_loss=criterion_reconstruct(decoded,labeled_data)
            class_loss = criterion_class(labeled_class, labels)
            total_loss=reconstruction_loss+class_loss
            preds = labeled_class.argmax(1).detach()
            train_corrects += (preds==labels.data).sum()
            # Backward pass
            total_loss.backward()
            optimizer.step()

            # unlabeled
            _,decoded = model(unlabeled_data)
            reconstruction_loss=criterion_reconstruct(decoded,unlabeled_data)
            reconstruction_loss.backward()
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
        _,data2=model(data)
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

