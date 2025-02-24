import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("/Users/max/MasterThesis/Training/"))
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
def get_datasets(dataname, fraction):
    if dataname == 'mnist34':
        print("This is the MNIST dataset with labels 3 and 4.")
        print("Imbalanced Ratio: ", fraction)
        # 数据预处理
        mnist34_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载 MNIST 数据集
        full_train_datasets = datasets.MNIST(root="/Users/max/MasterThesisData/MNIST", train=True, transform=mnist34_transforms, download=True)
        full_test_datasets = datasets.MNIST(root="/Users/max/MasterThesisData/MNIST", train=False, transform=mnist34_transforms, download=True)

        # 选取标签为 3 和 4 的索引
        indices3_train = [i for i in range(len(full_train_datasets)) if full_train_datasets.targets[i] == 3]
        indices4_train = [i for i in range(len(full_train_datasets)) if full_train_datasets.targets[i] == 4]

        indices3_test = [i for i in range(len(full_test_datasets)) if full_test_datasets.targets[i] == 3]
        indices4_test = [i for i in range(len(full_test_datasets)) if full_test_datasets.targets[i] == 4]

        # 获取训练集中标签为 3 和 4 的数据
        mnist3_train_data = full_train_datasets.data[indices3_train]
        mnist3_train_labels = torch.ones(len(indices3_train), dtype=torch.long)  # 标签 3 映射为 1 

        mnist4_train_data = full_train_datasets.data[indices4_train]
        mnist4_train_labels = torch.zeros(len(indices4_train), dtype=torch.long)  # 标签 4 映射为 0

        # 获取测试集中标签为 3 和 4 的数据
        mnist3_test_data = full_test_datasets.data[indices3_test]
        mnist3_test_labels = torch.ones(len(indices3_test), dtype=torch.long)  # 标签 3 映射为 1 

        mnist4_test_data = full_test_datasets.data[indices4_test]
        mnist4_test_labels = torch.zeros(len(indices4_test), dtype=torch.long)  # 标签 4 映射为 0

        # we can set the imbalanced ratio 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
        fraction = int(fraction * len(mnist3_train_data))  ### control the fraction of the data to be used
        selected_indices_4 = np.random.choice(len(mnist4_train_data), fraction, replace=False)

        fraction_mnist4_train_data = mnist4_train_data[selected_indices_4]
        fraction_mnist4_train_labels = mnist4_train_labels[selected_indices_4]

        # 创建最终的训练和测试数据集
        Final_train_data = torch.cat([mnist3_train_data, fraction_mnist4_train_data], dim=0)
        Final_train_labels = torch.cat([mnist3_train_labels, fraction_mnist4_train_labels], dim=0)

        Final_test_data = torch.cat([mnist3_test_data, mnist4_test_data], dim=0)
        Final_test_labels = torch.cat([mnist3_test_labels, mnist4_test_labels], dim=0)

        # 创建 TensorDataset
        Final_train_datasets = TensorDataset(Final_train_data.unsqueeze(1).float() / 255, Final_train_labels)
        Final_test_datasets = TensorDataset(Final_test_data.unsqueeze(1).float() / 255, Final_test_labels)

        # 数据加载器
        train_loader = DataLoader(Final_train_datasets, batch_size=64, shuffle=True)
        test_loader = DataLoader(Final_test_datasets, batch_size=64, shuffle=False)

        # 打印信息
        print("Number of label 3 in the final training set: ", len(mnist3_train_data))
        print("Number of label 4 in the final training set (after downsampling): ", len(fraction_mnist4_train_data))
        print("Number of label 3 in the final test set: ", len(mnist3_test_data))
        print("Number of label 4 in the final test set: ", len(mnist4_test_data))

        print("Total samples in final training set: ", len(Final_train_datasets))
        print("Total samples in final test set: ", len(Final_test_datasets))

        print("Number of batches in training set: ", len(train_loader))
        print("Number of batches in test set: ", len(test_loader))


        # 获取一个 batch
        images, labels = next(iter(train_loader))

        # 查看 Tensor 形状
        print(f"Images shape: {images.shape}")  # 形状为 (batch_size, channels, height, width)
        print(f"Labels shape: {labels.shape}")  # 形状为 (batch_size,)

        # 查看第一个样本的 Tensor 值
        print(f"First image tensor:\n{images[0]}")  # 打印第一个样本的 Tensor 数据
        print(f"First image label: {labels[0]}")  # 打印第一个样本的标签

        # 如果需要转换回 NumPy 并可视化：
        #import matplotlib.pyplot as plt

        # 转换为 NumPy 并显示
        plt.imshow(images[0].squeeze().numpy(), cmap="gray")
        plt.title(f"Label: {labels[0].item()}")
        plt.show()


        #import numpy as np

        # 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
        plt.figure(figsize=(20, 5)) 
        for i, imgs in enumerate(images[:20]):
            # 维度缩减
            npimg = np.squeeze(imgs.numpy())
            # 将整个figure分成2行10列，绘制第i+1个子图。
            plt.subplot(2, 10, i+1)
            plt.imshow(npimg, cmap=plt.cm.binary)
            plt.axis('off')


        X_train = []
        y_train = []

        for batch in train_loader:
            images, labels = batch  # images: (batch_size, 1, 28, 28), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                #print(f"Original shape: {img.shape}")  #  
                flattened_img = img.flatten()           
                #print(f"Flattened shape: {flattened_img.shape}")
                X_train.append(flattened_img) # 
                y_train.append(label)


        X_test = [] # features
        y_test = [] # labels
        for batch in test_loader:
            images, labels = batch  # images: (batch_size, 1,28  ,28 ), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                flattened_img = img.flatten()           # flatten the image
                X_test.append(flattened_img)
                y_test.append(label)


        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print("X_train.shape:", X_train.shape)
        print("y_train.shape:", y_train.shape)
        print("X_test.shape:", X_test.shape)
        print("y_test.shape:", y_test.shape)
        return X_train, y_train, X_test, y_test, train_loader, test_loader
    elif dataname == 'mnist17':
        print("This is the MNIST dataset with labels 1 and 7.")
        print("Imbalanced Ratio: ", fraction)
        # 数据预处理
        mnist17_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载 MNIST 数据集
        full_train_datasets = datasets.MNIST(root="/Users/max/MasterThesisData/MNIST", train=True, transform=mnist17_transforms, download=True)
        full_test_datasets = datasets.MNIST(root="/Users/max/MasterThesisData/MNIST", train=False, transform=mnist17_transforms, download=True)

        # 选取标签为 1 和 7 的索引
        indices1_train = [i for i in range(len(full_train_datasets)) if full_train_datasets.targets[i] == 1]
        indices7_train = [i for i in range(len(full_train_datasets)) if full_train_datasets.targets[i] == 7]

        indices1_test = [i for i in range(len(full_test_datasets)) if full_test_datasets.targets[i] == 1]
        indices7_test = [i for i in range(len(full_test_datasets)) if full_test_datasets.targets[i] == 7]

        # 获取训练集中标签为 1 和 7 的数据
        mnist1_train_data = full_train_datasets.data[indices1_train]
        mnist1_train_labels = torch.ones(len(indices1_train), dtype=torch.long)  # 标签 1 保持不变

        mnist7_train_data = full_train_datasets.data[indices7_train]
        mnist7_train_labels = torch.zeros(len(indices7_train), dtype=torch.long)  # 标签 7 映射为 0

        # 获取测试集中标签为 1 和 7 的数据
        mnist1_test_data = full_test_datasets.data[indices1_test]
        mnist1_test_labels = torch.ones(len(indices1_test), dtype=torch.long)  # 标签 1 保持不变

        mnist7_test_data = full_test_datasets.data[indices7_test]
        mnist7_test_labels = torch.zeros(len(indices7_test), dtype=torch.long)  # 标签 7 映射为 0

        # we can set the imbalanced ratio 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
        fraction = int(fraction * len(mnist1_train_data))  ### control the fraction of the data to be used
        selected_indices_7 = np.random.choice(len(mnist7_train_data), fraction, replace=False)

        fraction_mnist7_train_data = mnist7_train_data[selected_indices_7]
        fraction_mnist7_train_labels = mnist7_train_labels[selected_indices_7]

        # 创建最终的训练和测试数据集
        Final_train_data = torch.cat([mnist1_train_data, fraction_mnist7_train_data], dim=0)
        Final_train_labels = torch.cat([mnist1_train_labels, fraction_mnist7_train_labels], dim=0)

        Final_test_data = torch.cat([mnist1_test_data, mnist7_test_data], dim=0)
        Final_test_labels = torch.cat([mnist1_test_labels, mnist7_test_labels], dim=0)

        # 创建 TensorDataset
        Final_train_datasets = TensorDataset(Final_train_data.unsqueeze(1).float() / 255, Final_train_labels)
        Final_test_datasets = TensorDataset(Final_test_data.unsqueeze(1).float() / 255, Final_test_labels)

        # 数据加载器
        train_loader = DataLoader(Final_train_datasets, batch_size=64, shuffle=True)
        test_loader = DataLoader(Final_test_datasets, batch_size=64, shuffle=False)

        # 打印信息
        print("Number of label 1 in the final training set: ", len(mnist1_train_data))
        print("Number of label 7 in the final training set (after downsampling): ", len(fraction_mnist7_train_data))
        print("Number of label 1 in the final test set: ", len(mnist1_test_data))
        print("Number of label 7 in the final test set: ", len(mnist7_test_data))

        print("Total samples in final training set: ", len(Final_train_datasets))
        print("Total samples in final test set: ", len(Final_test_datasets))

        print("Number of batches in training set: ", len(train_loader))
        print("Number of batches in test set: ", len(test_loader))



        # 获取一个 batch
        images, labels = next(iter(train_loader))

        # 查看 Tensor 形状
        print(f"Images shape: {images.shape}")  # 形状为 (batch_size, channels, height, width)
        print(f"Labels shape: {labels.shape}")  # 形状为 (batch_size,)

        # 查看第一个样本的 Tensor 值
        print(f"First image tensor:\n{images[0]}")  # 打印第一个样本的 Tensor 数据
        print(f"First image label: {labels[0]}")  # 打印第一个样本的标签

        # 如果需要转换回 NumPy 并可视化：
        #import matplotlib.pyplot as plt

        # 转换为 NumPy 并显示
        plt.imshow(images[0].squeeze().numpy(), cmap="gray")
        plt.title(f"Label: {labels[0].item()}")
        plt.show()


        #import numpy as np

        # 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
        plt.figure(figsize=(20, 5)) 
        for i, imgs in enumerate(images[:20]):
            # 维度缩减
            npimg = np.squeeze(imgs.numpy())
            # 将整个figure分成2行10列，绘制第i+1个子图。
            plt.subplot(2, 10, i+1)
            plt.imshow(npimg, cmap=plt.cm.binary)
            plt.axis('off')


        X_train = []
        y_train = []

        for batch in train_loader:
            images, labels = batch  # images: (batch_size, 1, 28, 28), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                #print(f"Original shape: {img.shape}")  #  
                flattened_img = img.flatten()           
                #print(f"Flattened shape: {flattened_img.shape}")
                X_train.append(flattened_img) # 
                y_train.append(label)


        X_test = [] # features
        y_test = [] # labels
        for batch in test_loader:
            images, labels = batch  # images: (batch_size, 1,28  ,28 ), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                flattened_img = img.flatten()           # flatten the image
                X_test.append(flattened_img)
                y_test.append(label)


        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print("X_train.shape:", X_train.shape)
        print("y_train.shape:", y_train.shape)
        print("X_test.shape:", X_test.shape)
        print("y_test.shape:", y_test.shape)
        return X_train, y_train, X_test, y_test, train_loader, test_loader
    elif dataname == 'fashionmnist34':
        print("This is the FashionMNIST dataset with labels 3 and 4.")
        print("Imbalanced Ratio: ", fraction)
        # 数据预处理
        mnist34_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载 MNIST 数据集
        full_train_datasets = datasets.FashionMNIST(root="/Users/max/MasterThesisData/FashionMNIST", train=True, transform=mnist34_transforms, download=True)
        full_test_datasets = datasets.FashionMNIST(root="/Users/max/MasterThesisData/FashionMNIST", train=False, transform=mnist34_transforms, download=True)

        # 选取标签为 3 和 4 的索引
        indices3_train = [i for i in range(len(full_train_datasets)) if full_train_datasets.targets[i] == 3]
        indices4_train = [i for i in range(len(full_train_datasets)) if full_train_datasets.targets[i] == 4]

        indices3_test = [i for i in range(len(full_test_datasets)) if full_test_datasets.targets[i] == 3]
        indices4_test = [i for i in range(len(full_test_datasets)) if full_test_datasets.targets[i] == 4]

        # 获取训练集中标签为 3 和 4 的数据
        mnist3_train_data = full_train_datasets.data[indices3_train]
        mnist3_train_labels = torch.ones(len(indices3_train), dtype=torch.long)  # 标签 3 映射为 1 

        mnist4_train_data = full_train_datasets.data[indices4_train]
        mnist4_train_labels = torch.zeros(len(indices4_train), dtype=torch.long)  # 标签 4 映射为 0

        # 获取测试集中标签为 3 和 4 的数据
        mnist3_test_data = full_test_datasets.data[indices3_test]
        mnist3_test_labels = torch.ones(len(indices3_test), dtype=torch.long)  # 标签 3 映射为 1 

        mnist4_test_data = full_test_datasets.data[indices4_test]
        mnist4_test_labels = torch.zeros(len(indices4_test), dtype=torch.long)  # 标签 4 映射为 0
        # we can set the imbalanced ratio 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
        fraction = int(fraction * len(mnist3_train_data))  ### control the fraction of the data to be used
        selected_indices_4 = np.random.choice(len(mnist4_train_data), fraction, replace=False)
        fraction_mnist4_train_data = mnist4_train_data[selected_indices_4]
        fraction_mnist4_train_labels = mnist4_train_labels[selected_indices_4]


        # 创建最终的训练和测试数据集
        Final_train_data = torch.cat([mnist3_train_data, fraction_mnist4_train_data], dim=0)
        Final_train_labels = torch.cat([mnist3_train_labels, fraction_mnist4_train_labels], dim=0)

        Final_test_data = torch.cat([mnist3_test_data, mnist4_test_data], dim=0)
        Final_test_labels = torch.cat([mnist3_test_labels, mnist4_test_labels], dim=0)

        # 创建 TensorDataset
        Final_train_datasets = TensorDataset(Final_train_data.unsqueeze(1).float() / 255, Final_train_labels)
        Final_test_datasets = TensorDataset(Final_test_data.unsqueeze(1).float() / 255, Final_test_labels)

        # 数据加载器
        train_loader = DataLoader(Final_train_datasets, batch_size=64, shuffle=True)
        test_loader = DataLoader(Final_test_datasets, batch_size=64, shuffle=False)

        # 打印信息
        print("Number of label 3 in the final training set: ", len(mnist3_train_data))
        print("Number of label 4 in the final training set (after downsampling): ", len(fraction_mnist4_train_data))
        print("Number of label 3 in the final test set: ", len(mnist3_test_data))
        print("Number of label 4 in the final test set: ", len(mnist4_test_data))

        print("Total samples in final training set: ", len(Final_train_datasets))
        print("Total samples in final test set: ", len(Final_test_datasets))

        print("Number of batches in training set: ", len(train_loader))
        print("Number of batches in test set: ", len(test_loader))


        # 获取一个 batch
        images, labels = next(iter(train_loader))

        # 查看 Tensor 形状
        print(f"Images shape: {images.shape}")  # 形状为 (batch_size, channels, height, width)
        print(f"Labels shape: {labels.shape}")  # 形状为 (batch_size,)

        # 查看第一个样本的 Tensor 值
        print(f"First image tensor:\n{images[0]}")  # 打印第一个样本的 Tensor 数据
        print(f"First image label: {labels[0]}")  # 打印第一个样本的标签

        # 如果需要转换回 NumPy 并可视化：
        #import matplotlib.pyplot as plt

        # 转换为 NumPy 并显示
        plt.imshow(images[0].squeeze().numpy(), cmap="gray")
        plt.title(f"Label: {labels[0].item()}")
        plt.show()


        #import numpy as np

        # 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
        plt.figure(figsize=(20, 5)) 
        for i, imgs in enumerate(images[:20]):
            # 维度缩减
            npimg = np.squeeze(imgs.numpy())
            # 将整个figure分成2行10列，绘制第i+1个子图。
            plt.subplot(2, 10, i+1)
            plt.imshow(npimg, cmap=plt.cm.binary)
            plt.axis('off')


        X_train = []
        y_train = []

        for batch in train_loader:
            images, labels = batch  # images: (batch_size, 1, 28, 28), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                #print(f"Original shape: {img.shape}")  #  
                flattened_img = img.flatten()           
                #print(f"Flattened shape: {flattened_img.shape}")
                X_train.append(flattened_img) # 
                y_train.append(label)


        X_test = [] # features
        y_test = [] # labels
        for batch in test_loader:
            images, labels = batch  # images: (batch_size, 1,28  ,28 ), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                flattened_img = img.flatten()           # flatten the image
                X_test.append(flattened_img)
                y_test.append(label)


        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print("X_train.shape:", X_train.shape)
        print("y_train.shape:", y_train.shape)
        print("X_test.shape:", X_test.shape)
        print("y_test.shape:", y_test.shape)
        return X_train, y_train, X_test, y_test, train_loader, test_loader
    elif dataname == 'cifar10':
        print("This is the CIFAR10 dataset with labels 3 and 4.")
        print("Imbalanced Ratio: ", fraction)
        # CIFAR10 数据归一化参数
        # 数据预处理
        cifar10_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        # 加载 CIFAR10 数据集
        full_train_dataset = datasets.CIFAR10(
            root="/Users/max/MasterThesisData/CIFAR10", train=True, transform=cifar10_transforms, download=True)
        full_test_dataset = datasets.CIFAR10(
            root="/Users/max/MasterThesisData/CIFAR10", train=False, transform=cifar10_transforms, download=True)

        # 筛选训练集中的标签 3 和 4，并映射标签（标签 3 -> 1，标签 4 -> 0）
        indices3_train = [i for i in range(len(full_train_dataset)) if full_train_dataset.targets[i] == 3]
        indices4_train = [i for i in range(len(full_train_dataset)) if full_train_dataset.targets[i] == 4]

        # 筛选测试集中的标签 3 和 4，并映射标签
        indices3_test = [i for i in range(len(full_test_dataset)) if full_test_dataset.targets[i] == 3]
        indices4_test = [i for i in range(len(full_test_dataset)) if full_test_dataset.targets[i] == 4]

        # 训练集：获取标签为 3 的数据（映射为 1）
        cifar3_train_data = full_train_dataset.data[indices3_train]  # numpy 数组，形状 (n, 32, 32, 3)
        cifar3_train_labels = torch.ones(len(indices3_train), dtype=torch.long)

        # 训练集：获取标签为 4 的数据（映射为 0）
        cifar4_train_data = full_train_dataset.data[indices4_train]
        cifar4_train_labels = torch.zeros(len(indices4_train), dtype=torch.long)

        # 测试集：获取标签为 3 的数据（映射为 1）
        cifar3_test_data = full_test_dataset.data[indices3_test]
        cifar3_test_labels = torch.ones(len(indices3_test), dtype=torch.long)

        # 测试集：获取标签为 4 的数据（映射为 0）
        cifar4_test_data = full_test_dataset.data[indices4_test]
        cifar4_test_labels = torch.zeros(len(indices4_test), dtype=torch.long)

        # 控制不平衡比例，比如这里设定为 0.005（即只选取标签 4 数据中的一小部分）
        fraction = int(fraction * len(cifar3_train_data))
        selected_indices_4 = np.random.choice(len(cifar4_train_data), fraction, replace=False)
        fraction_cifar4_train_data = cifar4_train_data[selected_indices_4]
        fraction_cifar4_train_labels = cifar4_train_labels[selected_indices_4]

        # 拼接最终的训练和测试数据集
        Final_train_data = np.concatenate([cifar3_train_data, fraction_cifar4_train_data], axis=0)
        Final_train_labels = torch.cat([cifar3_train_labels, fraction_cifar4_train_labels], dim=0)

        Final_test_data = np.concatenate([cifar3_test_data, cifar4_test_data], axis=0)
        Final_test_labels = torch.cat([cifar3_test_labels, cifar4_test_labels], dim=0)

        # 将 numpy 数组转换为 tensor，并调整维度为 (N, C, H, W)
        Final_train_data = torch.from_numpy(Final_train_data).permute(0, 3, 1, 2).float()
        Final_test_data = torch.from_numpy(Final_test_data).permute(0, 3, 1, 2).float()

        # 归一化图像数据：先除以 255，再进行标准化
        # # 创建 TensorDataset
        # Final_train_datasets = TensorDataset(Final_train_data.unsqueeze(3).float() / 255, Final_train_labels)
        # Final_test_datasets = TensorDataset(Final_test_data.unsqueeze(3).float() / 255, Final_test_labels)

        # 创建 TensorDataset
        Final_train_dataset = TensorDataset(Final_train_data.float() / 255, Final_train_labels)
        Final_test_dataset = TensorDataset(Final_test_data.float() / 255, Final_test_labels)

        # 数据加载器
        train_loader = DataLoader(Final_train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(Final_test_dataset, batch_size=64, shuffle=False)

        # 打印数据信息
        print("Number of label 1 in the final training set: ", len(cifar3_train_data))
        print("Number of label 0 in the final training set (after downsampling): ", len(fraction_cifar4_train_data))
        print("Number of label 1 in the final test set: ", len(cifar3_test_data))
        print("Number of label 0 in the final test set: ", len(cifar4_test_data))

        print("Total samples in final training set: ", len(Final_train_dataset))
        print("Total samples in final test set: ", len(Final_test_dataset))

        print("Number of batches in training set: ", len(train_loader))
        print("Number of batches in test set: ", len(test_loader))


        
        ## 获取一个 batch
        images, labels = next(iter(train_loader))

        # 打印张量信息
        print(f"Images shape: {images.shape}")  # (batch_size, 3, 32, 32)
        print(f"Labels shape: {labels.shape}")

        # 将第一张图像的维度转换为 (H, W, C) 并显示
        img = images[0].permute(1, 2, 0).numpy()  # (32, 32, 3)
        plt.imshow(img)  # 对于彩色图像，无需指定 cmap
        plt.title(f"Label: {labels[0].item()}")
        plt.show()

        plt.figure(figsize=(20, 5))
        for i, img_tensor in enumerate(images[:20]):
            # 将每张图像转换为 (H, W, C)
            npimg = img_tensor.permute(1, 2, 0).numpy()
            plt.subplot(2, 10, i+1)
            plt.imshow(npimg)  # 彩色图像，不指定 cmap
            plt.axis('off')
        plt.show()


        X_train = []
        y_train = []

        for batch in train_loader:
            images, labels = batch  # images: (batch_size, 1, 28, 28), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                #print(f"Original shape: {img.shape}")  #  
                flattened_img = img.flatten()           
                #print(f"Flattened shape: {flattened_img.shape}")
                X_train.append(flattened_img) # 
                y_train.append(label)


        X_test = [] # features
        y_test = [] # labels
        for batch in test_loader:
            images, labels = batch  # images: (batch_size, 1,28  ,28 ), labels: (batch_size,)
            if hasattr(images, 'numpy'): # images is tensor
                images = images.numpy()
            # batch_size = 64
            for img, label in zip(images, labels):
                flattened_img = img.flatten()           # flatten the image
                X_test.append(flattened_img)
                y_test.append(label)


        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print("X_train.shape:", X_train.shape)
        print("y_train.shape:", y_train.shape)
        print("X_test.shape:", X_test.shape)
        print("y_test.shape:", y_test.shape)
        return X_train, y_train, X_test, y_test,train_loader, test_loader