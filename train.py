import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *

data_train=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor())
data_test=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

data_train_len=len(data_train)
data_test_len=len(data_test)

print("训练集的长度为： {}".format(data_train_len))
print("测试集的长度为： {}".format(data_test_len))

data_train_loader=DataLoader(data_train,batch_size=64)
data_test_loader=DataLoader(data_test,batch_size=64)

#创建网络模型
f_m=f_module()

#损失函数
loss_fn=nn.CrossEntropyLoss()

#学习率
learning_rate=1e-2

#优化器
optimizer=torch.optim.SGD(f_m.parameters(),lr=learning_rate)

#设置训练网络的参数
#记录训练的次数
total_train_step=0
#记录测试的次数
total_test_step=0
#训练的轮数
epoch=10

#添加tensorboard
writer=SummaryWriter("first_train")
start_time=time.time()

for i in range(epoch):
    print("-----------第{}轮训练开始------------".format(i+1))
    for data in data_train_loader:
        imgs,targets=data
        output=f_m(imgs)
        loss=loss_fn(output,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step=total_train_step+1
        if total_train_step%100==0:
            end_time = time.time()
            print("一轮运行时间为:{}".format(end_time - start_time))
            print("训练次数:{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train loss",loss.item(),total_train_step)

    #测试步骤
    #每次训练后进行评估
    total_test_loss=0
    with torch.no_grad():
        for data in data_test_loader:
            imgs,targets=data
            outputs=f_m(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test loss",total_test_loss,total_test_step)
    total_test_step=total_test_step+1

    torch.save(f_m,"f_m{}.pth".format(i))
    print("模型已保存")

writer.close()