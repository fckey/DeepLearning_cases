#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py    
@Contact :   sl.fang@qq.com
@License :   (C)Copyright 2021-2022, FangShaoLei

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/11/28 09:57   FangShaoLei  1.0         训练过程代码
"""
import time

import torch
from torch import nn
from model import AlexNet
import numpy as np
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

TRAIN_ROOT = r'data/train'
VALIDATE_ROOT = 'data/validate'
# 进行数据的处理，定义数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),       # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "validate": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
# 加载数据集
train_dataset = ImageFolder(TRAIN_ROOT, transform=data_transform['train'])
validate_dataset = ImageFolder(VALIDATE_ROOT, transform=data_transform['validate'])
# 讲数据进行小批量处理
train_dataloader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=0)
val_dataloader = DataLoader(validate_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AlexNet(num_classes=2).to(device)
# 定义一个损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    time_start = time.perf_counter()  # 对训练一个 epoch 计时
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        # 取最大的哪个坐标
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y==pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n+1
        # 打印训练进度（使训练过程可视化）
        rate = (batch + 1) / len(dataloader)  # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r{:^3.0f}%[{}->{}]".format(int(rate * 100), a, b), end="")
    print('%f s' % (time.perf_counter() - time_start))
    # 返回平均的loss
    train_loss = loss / n
    train_acc = current / n
    print('train_loss' + str(train_loss))
    print('train_acc' + str(train_acc))
    return train_loss, train_acc

# 定义一个验证函数
def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss' + str(val_loss))
    print('val_acc' + str(val_acc))
    return val_loss, val_acc

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()

# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []


epoch = 20
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t+1}\n-----------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重
    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = val_acc
        print(f"save best model, 第{t+1}轮")
        torch.save(model.state_dict(), 'save_model/best_model.pth')
    # 保存最后一轮的权重文件
    if t == epoch-1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('Done!')
