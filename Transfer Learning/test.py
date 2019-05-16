from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode打开交互模式
#数据集字典

#图像剪裁224
#以0.5的概率水平翻转给定的PIL图像
#维度转换，类似于RGB转BGR
#规范化，给定均值：(R,G,B)方差：（R，G，B），将会把Tensor正则化。即：Normalized_image=(image-mean)/std
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),#训练集
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),#验证集
}
"""
ImageFolder(root,transform=None,target_transform=None,loader=
default_loader)
root : 在指定的root路径下面寻找图片 
transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象 
target_transform :对label进行变换 
loader: 指定加载图片的函数，默认操作是读取PIL image对象
"""
data_dir = 'data/hymenoptera_data'#数据文件路径
"""
将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入

"""
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
#调用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#可视化一些训练图像
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))#矩阵维度调换
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean#矩阵星乘，点乘求内积
    inp = np.clip(inp, 0, 1)

#数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 #a_max，小于a_min,的就使得它等于a_min。
#numpy.clip(a, a_min, a_max, out=None)[source]

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated时延

# Get a batch of training data
#iter：迭代器
#next：返回迭代器的下一个项目
inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)#将多个图像频接成一个图像。
#imshow(out, title=[class_names[x] for x in classes])#显示图像。
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()#返回当前时间

    best_model_wts = copy.deepcopy(model.state_dict())#数据完整复制
    best_acc = 0.0
#num_epochs循环次数
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))#format格式化函数，代替%
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()#学习率相关。。
                model.train()  # Set model to training mode训练
            else:
                model.eval()   # Set model to evaluate mode测试

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients0参数梯度
                optimizer.zero_grad()

                # forward#前向传播
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)#传入模型
                    _, preds = torch.max(outputs, 1)#返回输出最大值
                    loss = criterion(outputs, labels)#计算交叉熵

                    # backward + optimize only if in training phase反向传播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics计算训练时正确率
                running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model计算最终正确率。
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since#计算时间
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)#加载模型
    return model
    
#模型可视化
def visualize_model(model, num_images=6):
    was_training = model.training#训练方法
    model.eval()#测试方法
    images_so_far = 0
    fig = plt.figure()#绘图
#不求导no_grad
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):#枚举
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)#画图
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
#加载预训练权重
model_ft = models.resnet18(pretrained=True)#构造一个ResNet-18模型
#progress（bool） - 如果为True，则显示下载到stderr的进度条
num_ftrs = model_ft.fc.in_features#定义输入数量
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)#选择硬件，CPU or GPU

criterion = nn.CrossEntropyLoss()#定义代价函数
#随机梯度下降，momentum冲量，优化”山谷“的速率
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#学习率衰减，指数衰减。
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
#训练模型
#可视化模型
visualize_model(model_ft)

#ConvNet 作为固定特征提取器,parameters参数
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as最终层被优化
# opposed to before.反向传播
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs学习率衰减
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#训练
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
#显示图像。
visualize_model(model_conv)
plt.ioff()
plt.show()