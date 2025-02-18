import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2  # 二分类任务

class BinaryCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取网络
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 第一层卷积,卷积核大小为3*3
        self.pool1 = nn.MaxPool2d(2)                  # 设置池化层，池化核大小为2*2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 第二层卷积,卷积核大小为3*3   
        self.pool2 = nn.MaxPool2d(2) 
                                      
        # 分类网络
        self.fc1 = nn.Linear(1600, 64)          
        self.fc2 = nn.Linear(64, num_classes)  # 修改输出层维度为 2

    # 前向传播
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))     
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 直接输出 logits，不加激活函数

        return x

if __name__ == '__main__':
    model = BinaryCNN1()
    print(model)