import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            tconv5 = self.tconv6(tconv5)
            output = tconv5
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes


class _netG_CIFAR10(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            output = tconv5
        return output


class _netD_CIFAR10(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes
    



# 定义用于MNIST生成器的网络
class _netG_MNIST(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG_MNIST, self).__init__()  # 调用父类的构造函数
        self.ngpu = ngpu  # 保存GPU数量
        self.nz = nz      # 噪声向量的维度（例如：噪声+标签拼接后为110）

        # 全连接层：将输入噪声向量映射为256*7*7个特征
        self.fc = nn.Linear(nz, 256 * 7 * 7)  # 输入维度为nz，输出维度为256*7*7

        # 转置卷积层1：将256通道、7x7的特征图转换为128通道、14x14的特征图
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 转置卷积，尺寸放大2倍
            nn.BatchNorm2d(128),         # 对128个通道做批归一化
            nn.ReLU(True)                # 使用ReLU激活函数
        )

        # 转置卷积层2：将128通道、14x14的特征图转换为1通道、28x28的图像
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 转置卷积，尺寸放大2倍
            nn.Tanh()                  # 使用Tanh激活函数，将输出映射到[-1,1]之间
        )

    def forward(self, input):
        # 若使用CUDA且多GPU，则采用数据并行方式
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)                           # 将输入展平为二维张量[batch_size, nz]
            fc = nn.parallel.data_parallel(self.fc, input, range(self.ngpu))  # 并行通过全连接层
            fc = fc.view(-1, 256, 7, 7)                               # 重塑为[batch_size, 256, 7, 7]的特征图
            tconv1 = nn.parallel.data_parallel(self.tconv1, fc, range(self.ngpu))  # 并行通过第一个转置卷积层
            tconv2 = nn.parallel.data_parallel(self.tconv2, tconv1, range(self.ngpu))  # 并行通过第二个转置卷积层
            output = tconv2                                         # 得到生成的图像
        else:
            input = input.view(-1, self.nz)                           # 将输入展平为二维张量[batch_size, nz]
            fc = self.fc(input)                                       # 通过全连接层
            fc = fc.view(-1, 256, 7, 7)                               # 重塑为[batch_size, 256, 7, 7]的特征图
            tconv1 = self.tconv1(fc)                                  # 通过第一个转置卷积层
            tconv2 = self.tconv2(tconv1)                              # 通过第二个转置卷积层
            output = tconv2                                         # 得到生成的图像
        return output  # 返回生成的28x28单通道图像

# 定义用于MNIST判别器的网络
class _netD_MNIST(nn.Module):
    def __init__(self, ngpu, num_classes=2):
        super(_netD_MNIST, self).__init__()  # 调用父类的构造函数
        self.ngpu = ngpu  # 保存GPU数量

        # 卷积层1：将输入1通道图像转换为16通道，输出尺寸为14x14（28x28经步长为2降采样）
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 卷积操作
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数，负半轴斜率为0.2
            nn.Dropout(0.5, inplace=False)      # Dropout防止过拟合
        )

        # 卷积层2：保持空间尺寸不变，增加通道数到32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 卷积操作
            nn.BatchNorm2d(32),             # 对32个通道做批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Dropout(0.5, inplace=False)      # Dropout防止过拟合
        )

        # 卷积层3：将特征图尺寸减半（14x14→7x7），通道数增至64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 卷积操作
            nn.BatchNorm2d(64),             # 对64个通道做批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Dropout(0.5, inplace=False)      # Dropout防止过拟合
        )

        # 卷积层4：保持7x7尺寸，通道数增至128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),  # 卷积操作
            nn.BatchNorm2d(128),            # 对128个通道做批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Dropout(0.5, inplace=False)      # Dropout防止过拟合
        )

        # 卷积层5：将特征图尺寸减半（7x7→4x4，因7+2-3=6，6//2+1=4），通道数增至256
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 卷积操作
            nn.BatchNorm2d(256),            # 对256个通道做批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Dropout(0.5, inplace=False)      # Dropout防止过拟合
        )

        # 卷积层6：保持特征图尺寸为4x4，通道数增至512
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),  # 卷积操作
            nn.BatchNorm2d(512),            # 对512个通道做批归一化
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Dropout(0.5, inplace=False)      # Dropout防止过拟合
        )

        # 全连接层：用于判别图像的真伪（discriminator）
        self.fc_dis = nn.Linear(4 * 4 * 512, 1)  # 将4x4x512的特征展平后映射到1个输出
        # 全连接层：用于辅助分类器，预测图像类别
        self.fc_aux = nn.Linear(4 * 4 * 512, num_classes)  # 将特征映射到类别数
        # Softmax层：用于计算辅助分类器输出的概率分布
        self.softmax = nn.Softmax(dim=1)  # 指定在类别维度上做归一化
        # Sigmoid层：用于计算判别器输出的真伪概率
        self.sigmoid = nn.Sigmoid()  # 将输出映射到[0,1]之间

    def forward(self, input):
        # 若使用CUDA且多GPU，则采用数据并行方式
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))  # 第一卷积层并行计算
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))  # 第二卷积层并行计算
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))  # 第三卷积层并行计算
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))  # 第四卷积层并行计算
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))  # 第五卷积层并行计算
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))  # 第六卷积层并行计算
            flat6 = conv6.view(-1, 4 * 4 * 512)  # 将卷积输出展平成二维张量
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))  # 判别器全连接层并行计算
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))  # 辅助分类器全连接层并行计算
        else:
            conv1 = self.conv1(input)       # 通过第一卷积层
            conv2 = self.conv2(conv1)       # 通过第二卷积层
            conv3 = self.conv3(conv2)       # 通过第三卷积层
            conv4 = self.conv4(conv3)       # 通过第四卷积层
            conv5 = self.conv5(conv4)       # 通过第五卷积层
            conv6 = self.conv6(conv5)       # 通过第六卷积层
            flat6 = conv6.view(-1, 4 * 4 * 512)  # 将卷积输出展平成二维张量
            fc_dis = self.fc_dis(flat6)     # 通过判别器全连接层
            fc_aux = self.fc_aux(flat6)     # 通过辅助分类器全连接层

        classes = self.softmax(fc_aux)  # 计算辅助分类器的类别概率分布
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)  # 计算判别器的真伪概率，并调整输出形状
        return realfake, classes  # 返回真伪判断结果和类别预测