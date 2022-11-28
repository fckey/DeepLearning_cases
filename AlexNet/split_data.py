#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   split_data.py    
@Contact :   sl.fang@qq.com
@License :   (C)Copyright 2021-2022, FangShaoLei

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/11/28 09:38   FangShaoLei  1.0        对原来的数据集进行分割，分割成测试集和训练集
"""
import os
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
# 获取所有要分类的文件夹
file_path = "./raw_data/"
train_path = 'data/train/'
validate_path = 'data/validate/'
# 列出所有花的种类
flow_cases  = [clazz for clazz in os.listdir(file_path)]
# 创建出验证集和训练集文件夹，并由类名在其目录下创建五个子目录
mkfile(train_path)
mkfile(validate_path)
for clazz in flow_cases:
    mkfile(train_path + clazz)
    mkfile(validate_path + clazz)
# 按照比例来进行划分, 训练集和测试集 = 9 ： 1
split_rate = 0.1
# 遍历所有类别的全部图像，并按照比例分成训练集合验证集
for clazz in flow_cases:
    clazz_path = file_path + '/' + clazz + '/' # 某一个类别的文件夹
    images = os.listdir(clazz_path) # images 列表存储来目录下的所有图片的名称
    num = len(images)
    sample_images = random.sample(images, k=int(num * split_rate)) # 从images列表随机sample出k个样本
    for index, image in enumerate(images):
        # sample_images保存的是所有取出来的图片
        if image in sample_images:
            image_path = clazz_path + image
            new_path = validate_path + clazz
            copy(image_path, new_path)
        # 其他的所有图片都保留在训练集中
        else:
            image_path = clazz_path + image
            new_path = train_path + clazz
            copy(image_path, new_path)
        print(f'\r[{clazz}] processing [{index + 1} / {num}]', end="") # process bar
    print()

print("processing done!")



