#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predict.py    
@Contact :   sl.fang@qq.com
@License :   (C)Copyright 2021-2022, FangShaoLei

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/11/28 10:26   FangShaoLei  1.0         验证函数
"""
import torch
from model import AlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


TRAIN_ROOT = 'data/train'
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

# 加载模型
model.load_state_dict(torch.load("save_model/best_model.pth", map_location='cpu'))
# 获取预测结果
classes = [
    "cat",
    "dog",
]

# 把张量转化为照片格式
show = ToPILImage()

# 进入到验证阶段
model.eval()
for i in range(10):
    x, y = validate_dataset[i][0], validate_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        # 用argmax获取概率最大的一个物体
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')