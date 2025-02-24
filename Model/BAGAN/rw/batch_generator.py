"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import numpy as np  # 导入NumPy库，用于数组操作
from torchvision.datasets import MNIST, CIFAR10  # 从torchvision导入MNIST和CIFAR10数据集
import os  # 导入os模块，用于文件操作
import torch  # 导入PyTorch，用于后续可能的张量操作

class BatchGenerator:
    # 定义常量，表示训练集和测试集标识
    TRAIN = 1  # 训练数据标识
    TEST = 0   # 测试数据标识

    def __init__(self, data_src, batch_size=32, class_to_prune=None, unbalance=0, dataset='MNIST'):
        # 初始化函数，data_src: 数据来源（TRAIN或TEST），batch_size: 批次大小，
        # class_to_prune: 待剪枝的类别（若不为None，则对该类别进行数据下采样），
        # unbalance: 不平衡因子，dataset: 数据集名称（'MNIST'或'CIFAR10'）
        assert dataset in ('MNIST', 'CIFAR10'), 'Unknown dataset: ' + dataset  # 检查数据集名称是否合法
        self.batch_size = batch_size  # 保存批次大小
        self.data_src = data_src  # 保存数据来源标识

        # 加载数据
        if dataset == 'MNIST':
            # 如果使用MNIST数据集，则调用torchvision.datasets.MNIST加载数据
            if self.data_src == self.TEST:
                mnist_data = MNIST(root="dataset/mnist", train=False, download=True)  # 加载测试集
            else:
                mnist_data = MNIST(root="dataset/mnist", train=True, download=True)   # 加载训练集

            # 确保批次大小大于0
            assert self.batch_size > 0, 'Batch size has to be a positive integer!'

            # 获取图像数据，mnist_data.data的形状为[N, 28, 28]，类型为uint8
            self.dataset_x = mnist_data.data.numpy()  # 转换为NumPy数组，形状[N, 28, 28]
            # 将图像数据转换为浮点型，并归一化到[0,1]
            self.dataset_x = self.dataset_x.astype(np.float32) / 255.0  
            # 将数据归一化到[-1,1]区间
            self.dataset_x = (self.dataset_x - 0.5) * 2.0  
            # 为数据添加通道维度，将形状转换为[N, 1, 28, 28]
            self.dataset_x = np.expand_dims(self.dataset_x, axis=1)  

            # 获取标签数据，mnist_data.targets为PyTorch张量，转换为NumPy数组
            self.dataset_y = mnist_data.targets.numpy()

        elif dataset == 'CIFAR10':
            # 如果使用CIFAR10数据集，则调用torchvision.datasets.CIFAR10加载数据
            if self.data_src == self.TEST:
                cifar_data = CIFAR10(root="dataset/cifar10", train=False, download=True)  # 加载测试集
            else:
                cifar_data = CIFAR10(root="dataset/cifar10", train=True, download=True)   # 加载训练集

            # 获取图像数据，cifar_data.data的形状为[N, H, W, C]，类型为uint8
            self.dataset_x = cifar_data.data  # 形状[N, 32, 32, 3]
            # 将数据转换为浮点型
            self.dataset_x = self.dataset_x.astype(np.float32)
            # 调整数据顺序为通道优先格式，将形状转换为[N, C, H, W]
            self.dataset_x = np.transpose(self.dataset_x, axes=(0, 3, 1, 2))
            # 将数据归一化到[-1,1]区间，公式：(x-127.5)/127.5
            self.dataset_x = (self.dataset_x - 127.5) / 127.5

            # 获取标签数据，cifar_data.targets为列表，将其转换为NumPy数组
            self.dataset_y = np.array(cifar_data.targets)

        # 检查图像和标签数量是否匹配
        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # 计算每个类别的样本数量
        classes = np.unique(self.dataset_y)  # 获取所有唯一的类别标签
        self.classes = classes  # 保存类别标签数组
        per_class_count = []  # 用于存储每个类别的样本数
        for c in classes:
            per_class_count.append(np.sum(self.dataset_y == c))  # 统计类别c的样本数量

        # 如果指定了待剪枝的类别，则对该类别的数据进行下采样
        if class_to_prune is not None:
            # 获取所有样本的索引
            all_ids = np.arange(len(self.dataset_x))
            # 构造布尔掩码，选出标签等于class_to_prune的样本
            mask = (self.dataset_y == class_to_prune)
            # 根据掩码筛选出该类别的所有索引
            all_ids_c = all_ids[mask]
            # 打乱这些索引的顺序
            np.random.shuffle(all_ids_c)

            # 计算其他类别的样本数量
            other_class_count = np.array(per_class_count)
            # 删除待剪枝类别对应的计数
            other_class_count = np.delete(other_class_count, class_to_prune)
            # 计算待保留的样本数量，取其他类别中最大样本数乘以不平衡因子
            to_keep = int(np.ceil(unbalance * np.max(other_class_count)))
            # 需要删除的索引为该类别中超过to_keep部分的索引
            to_delete = all_ids_c[to_keep:]
            # 从数据集中删除这些样本（沿着样本维度删除）
            self.dataset_x = np.delete(self.dataset_x, to_delete, axis=0)
            self.dataset_y = np.delete(self.dataset_y, to_delete, axis=0)

        # 剪枝后重新统计每个类别的样本数量
        per_class_count = []
        for c in classes:
            per_class_count.append(np.sum(self.dataset_y == c))
        self.per_class_count = per_class_count  # 保存每个类别的样本数列表

        # 构建标签表，假设类别总数为10
        self.label_table = [str(c) for c in range(10)]

        # 将所有标签预加载到self.labels中
        self.labels = self.dataset_y.copy()

        # 构建每个类别对应的样本索引字典
        self.per_class_ids = {}  # 初始化字典
        ids = np.arange(len(self.dataset_x))  # 获取所有样本的索引数组
        for c in classes:
            # 保存类别c对应的索引，利用布尔索引筛选
            self.per_class_ids[c] = ids[self.labels == c]

    def get_samples_for_class(self, c, samples=None):
        # 获取类别c的若干样本，samples默认为批次大小
        if samples is None:
            samples = self.batch_size  # 如果未指定样本数，则使用批次大小
        # 随机打乱类别c的样本索引
        np.random.shuffle(self.per_class_ids[c])
        # 取出前samples个样本的索引
        to_return = self.per_class_ids[c][0:samples]
        # 返回对应样本数据
        return self.dataset_x[to_return]

    def get_label_table(self):
        # 返回标签表（类别名称列表）
        return self.label_table

    def get_num_classes(self):
        # 返回类别总数
        return len(self.label_table)

    def get_class_probability(self):
        # 返回每个类别的概率（样本数量占总样本数的比例）
        return np.array(self.per_class_count) / np.sum(self.per_class_count)

    # 以下函数用于数据访问和获取数据形状信息
    def get_num_samples(self):
        # 返回数据集中样本总数
        return self.dataset_x.shape[0]

    def get_image_shape(self):
        # 返回单个图像的形状，格式为[通道数, 高度, 宽度]
        return [self.dataset_x.shape[1], self.dataset_x.shape[2], self.dataset_x.shape[3]]

    def next_batch(self):
        # 定义生成器，按批次返回数据和标签
        dataset_x = self.dataset_x  # 获取图像数据
        labels = self.labels        # 获取标签数据

        # 构造样本索引数组
        indices = np.arange(dataset_x.shape[0])
        # 随机打乱索引顺序
        np.random.shuffle(indices)

        # 按批次遍历整个数据集
        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            # 获取当前批次的索引
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            # 对索引进行排序（可选操作，以保证数据顺序一致）
            access_pattern = sorted(access_pattern)
            # 使用生成器返回当前批次的图像和对应的标签
            yield dataset_x[access_pattern, :, :, :], labels[access_pattern]
