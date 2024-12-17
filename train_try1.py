from main import *

# 这个是拿来尝试的，效果不咋样，不用写在报告里面

net2=Net2().to(device)
flip_dataset=FlippedDataset(train_dataset)
flip_loader_=DataLoader(flip_dataset,batch_size=batch_size,shuffle=True)
enhanced_dataset2=EnhancedDataset(train_dataset,offset=2)
enhance_loader_2=DataLoader(enhanced_dataset2,batch_size=batch_size,shuffle=True)
enhanced_dataset_1=EnhancedDataset(train_dataset,offset=1)
enhance_loader_1=DataLoader(enhanced_dataset_1,batch_size=batch_size,shuffle=True)
unlabeled_loader=DataLoader(unlabeled_dataset,batch_size=batch_size,shuffle=True)



recurse=3

def train_(epochs=epochs,loader=flip_loader_,get_optimizer=get_optimizer,loss_fn=loss_fn,recurse=recurse):
    net2.train()
    for i in range(epochs):
        corrects=0
        optimizer=get_optimizer(net2)
        for inputs, labels in tqdm(loader,leave=False):
            optimizer.zero_grad()
            pre=torch.ones(inputs.shape[0],10).to(device)/10
            outputs = net2(inputs,pre)

            for i in range(recurse-1):
                outputs=net2(inputs,outputs)
            loss:torch.Tensor = loss_fn(outputs,labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
        print("acc",(corrects / len(loader.dataset)).item())

def test(recurse=recurse,use_softmax=True,test_loader=test_loader):
    
    net2.eval()
    def do_run(inputs):
        outputs=torch.ones(inputs.shape[0],10).to(device)/10
        for _ in range(recurse):
            outputs = net2(inputs,outputs)
        return F.softmax(outputs,dim=-1) if use_softmax else outputs
    corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, leave=False):
            outputs=do_run(inputs)
            for i in transform_offset(inputs,offset=1):
                outputs+=do_run(i)
            preds = outputs.argmax(1).detach()
            corrects += (preds==labels.data).sum()
    return (corrects / len(test_loader.dataset)).item()
def semi_supervised_training_with_regularization(unlabeled_dataloader=unlabeled_loader, labeled_dataloader=train_loader, model=net2, criterion=loss_fn, optimizer_getter=get_optimizer, num_epochs=epochs, lambda_l2=0.01):
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
    def do_run(inputs):
        outputs=torch.ones(inputs.shape[0],10).to(device)/10
        for _ in range(recurse):
            outputs = net2(inputs,outputs)
        return outputs
    for epoch in range(num_epochs):
        # 为无标签的数据生成伪标签并进行训练
        for data,(x,y) in zip(unlabeled_dataloader,labeled_dataloader):
            optimizer.zero_grad()
            outputs:torch.Tensor = do_run(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            with torch.no_grad():  # 不计算生成伪标签的梯度
                pseudo_labels = do_run(data).argmax(dim=1)  # 生成伪标签
            outputs = do_run(data)
            loss = criterion(outputs, pseudo_labels)  # 使用伪标签计算损失
            
            # 添加L2正则化
            l2_reg = torch.tensor(0.).to(data.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += lambda_l2 * l2_reg
            
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}] completed')

highest_acc=0
highest_acc_data=None
def update_final_output():
    global highest_acc,highest_acc_data
    net.eval()
    test_acc=test()
    if test_acc>highest_acc:
        print("new higher test_acc",test_acc)
    else:
        print("test_acc not higher",test_acc,"highest_acc",highest_acc)
        return
    def do_run(inputs):
        outputs=torch.ones(inputs.shape[0],10).to(device)/10
        for _ in range(recurse):
            outputs = net2(inputs,outputs)
        return F.softmax(outputs,dim=-1) 
    with torch.no_grad():
        x=torch.stack([final_dataset[i] for i in range(len(final_dataset))])
        y=do_run(x)
        for i in transform_offset(x):
            y+=do_run(i)
        y=y.argmax(1).detach().cpu().unsqueeze(0)
        y=y.numpy()
    
    highest_acc=test_acc
    highest_acc_data=y
    

def save_final():
    import os
    os.makedirs("out2",exist_ok=True)
    print("shape",highest_acc_data.shape)
    np.save(f"out2/output_{highest_acc}.npy",highest_acc_data)

def train(epochs=15):
    dataloaders=[flip_loader_,enhance_loader_2,enhance_loader_1]
    for i in range(epochs):
        semi_supervised_training_with_regularization(num_epochs=1)
        for data_loader in dataloaders:
                train_(epochs=1,loader=data_loader)
                update_final_output()

if __name__=="__main__":
    train(114514)