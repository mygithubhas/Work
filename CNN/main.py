# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
#把多个PIL图像转化为tensor，给定均值（RGB）方差（RGB）
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#下载CIFAR10数据集，root主目录train训练集download显示下载，transform输入转换函数
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#使用多线程，批次大小，用于数据加载的子进程数。在每个批次重新调整数据0表示数据将加载到主进程中
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# 输出图像的函数


def imshow(img):
    img = img / 2 + 0.5     # unnormalize非标准化
    npimg = img.numpy()#转化为numpy类型
    plt.imshow(np.transpose(npimg, (1, 2, 0)))#置换数组维度
    plt.show()

# 随机得到一些训练图片
dataiter = iter(trainloader)#生成迭代器iter(object[, sentinel])，object -- 支持迭代的集合对象。sentinel -- 如果传递了第二个参数，则参数 object 必须是一个可调用的对象
images, labels = dataiter.next()#next() 返回迭代器的下一个项目。
# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))#F 是import torch.nn.functional as F
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
#net.to(device)




import torch.optim as optim

criterion = nn.CrossEntropyLoss()#nn.logSoftmax()和nn.NLLLoss()的整合，非传统意义交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#lr学习率，momentum学习率衰减？parameters()？？随机梯度下降法
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        #梯度计算好后进行单次优化，在backward()之后
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
#第几轮，第几次
print('Finished Training')



dataiter = iter(testloader)
images, labels = dataiter.next()
#inputs, labels = inputs.to(device), labels.to(device)

# 输出拼接后的图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)#返回每一行中最大值的那个元素，且返回其索引（0列，1行）

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))#join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

correct = 0
total = 0
with torch.no_grad():#全局不追踪
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)#返回最大值
        total += labels.size(0)
        correct += (predicted == labels).sum().item()#求和输出

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))#输出正确率

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()#判断两个张量是否相等，相等去掉维度为1的
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


