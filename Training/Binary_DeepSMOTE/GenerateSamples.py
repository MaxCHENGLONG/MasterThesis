# -*- coding: utf-8 -*-  # 指定源代码文件的编码为UTF-8，确保中文字符正常显示

import collections  # 导入collections模块，用于计数和其他容器数据结构操作
import torch  # 导入PyTorch库，用于构建和训练深度学习模型
import torch.nn as nn  # 从torch中导入神经网络模块，便于构建各类神经网络层
import numpy as np  # 导入NumPy库，用于高效的数组运算和数值计算
from sklearn.neighbors import NearestNeighbors  # 从scikit-learn中导入最近邻算法，用于生成SMOTE样本
import os  # 导入os模块，用于文件和目录操作

print(torch.version.cuda)  # 打印当前PyTorch使用的CUDA版本，如输出"10.1"

import time  # 导入time模块，用于记录和计算时间
t0 = time.time()  # 记录程序开始时的时间，用于后续计算整个程序的运行时长

##############################################################################
"""args for models"""
# 设置用于模型构建和训练的参数

args = {}  # 创建一个空字典，用于存放所有的参数
args['dim_h'] = 64          # 隐藏层基准通道数，用于控制卷积层中隐藏层的规模
args['n_channel'] = 1       # 输入数据的通道数，这里设为1（灰度图像），若为3则为彩色图

args['n_z'] = 300 #600     # 潜在空间（编码器输出）的维度，这里设为300

args['sigma'] = 1.0        # 潜在空间中使用的方差参数，可用于模型正则化（备用参数）
args['lambda'] = 0.01      # 判别器损失权重的超参数（可能用于对抗训练）
args['lr'] = 0.0002        # Adam优化器的学习率，决定参数更新的步长
args['epochs'] = 1 #50         # 训练轮数，这里设为1轮（调试时设少一点，正式训练时可增至50或更多）
args['batch_size'] = 100   # 每个训练批次的样本数
args['save'] = True        # 是否在每个epoch结束后保存模型权重
args['train'] = True       # 是否执行训练过程，若为False则加载已有模型
args['dataset'] = 'mnist'  # 指定所用数据集，这里设为'mnist'，也可选择'fmnist'等

##############################################################################
## create encoder model and decoder model
# 定义编码器（Encoder）模型和解码器（Decoder）模型

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()  # 调用父类(nn.Module)的构造函数
        self.n_channel = args['n_channel']  # 获取输入数据的通道数
        self.dim_h = args['dim_h']          # 获取隐藏层基准通道数
        self.n_z = args['n_z']              # 获取潜在空间的维度

        # 定义一系列卷积层，用于提取图像特征
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),  
            # 第一层卷积：输入通道数为n_channel，输出通道数为dim_h，卷积核大小4，步幅2，填充1，不使用偏置
            # nn.ReLU(True),  # 可选ReLU激活（已注释）
            nn.LeakyReLU(0.2, inplace=True),  # 使用LeakyReLU激活函数，负半部分斜率0.2
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),  
            # 第二层卷积：通道数从dim_h增加到dim_h*2
            nn.BatchNorm2d(self.dim_h * 2),  # 对第二层卷积输出进行批归一化
            # nn.ReLU(True),  # 可选ReLU激活（已注释）
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),  
            # 第三层卷积：通道数从dim_h*2增加到dim_h*4
            nn.BatchNorm2d(self.dim_h * 4),  # 批归一化
            # nn.ReLU(True),  # 可选激活（已注释）
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数

            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),  
            # 第四层卷积：通道数从dim_h*4增加到dim_h*8
            # 下面是备用配置（注释掉的3d和32x32的卷积配置）
            # nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),  # 对第四层卷积输出进行批归一化，注释中提到“40 X 8 = 320”
            # nn.ReLU(True),  # 可选激活（已注释）
            nn.LeakyReLU(0.2, inplace=True)   # 使用LeakyReLU激活函数
            # 注释掉的备用层：可能用于调整输出尺寸，如使用nn.Conv2d(self.dim_h * 8, 1, 2, 1, 0, bias=False)
        )
        # 最后一层为全连接层，将卷积层输出映射到潜在空间
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
        # 注意：这里dim_h * (2 ** 3)等价于dim_h * 8，假定卷积层输出的特征维度为该值

    def forward(self, x):
        # 前向传播函数
        # print('enc')  # 调试用，打印“enc”
        # print('input ', x.size())  # 可打印输入张量的尺寸（例如[batch, channel, H, W]）
        x = self.conv(x)  # 将输入x通过卷积层提取特征
        # print('aft conv ', x.size())  # 调试用，打印经过卷积层后的尺寸
        x = x.squeeze()  # 压缩张量，移除尺寸为1的维度（例如将[batch, 320, 1, 1]压缩为[batch, 320]）
        # print('aft squeeze ', x.size())  # 调试用，打印压缩后的尺寸
        x = self.fc(x)  # 将压缩后的特征通过全连接层映射到潜在空间维度
        # print('out ', x.size())  # 调试用，打印最终输出尺寸，应为[batch, n_z]
        return x  # 返回编码后的潜在向量

# 定义解码器模型，用于将潜在向量还原成图像
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()  # 调用父类构造函数
        self.n_channel = args['n_channel']  # 获取输出图像的通道数
        self.dim_h = args['dim_h']          # 获取隐藏层基准通道数
        self.n_z = args['n_z']              # 获取潜在向量的维度

        # 定义全连接层，将潜在向量扩展为足够还原为卷积特征图的尺寸
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),  # 将潜在向量映射到大小为dim_h*8*7*7的特征向量
            nn.ReLU()  # 使用ReLU激活函数
        )

        # 定义反卷积层（转置卷积），用于将全连接层的输出还原为图像
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),  
            # 第一层反卷积：将通道数从dim_h*8降到dim_h*4，卷积核大小为4
            nn.BatchNorm2d(self.dim_h * 4),  # 批归一化
            nn.ReLU(True),  # 激活函数
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),  
            # 第二层反卷积：通道数从dim_h*4降到dim_h*2
            nn.BatchNorm2d(self.dim_h * 2),  # 批归一化
            nn.ReLU(True),  # 激活函数
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),  
            # 第三层反卷积：将通道数降为1，同时上采样，恢复图像尺寸
            # nn.Sigmoid()  # 可选Sigmoid激活，使输出在[0,1]之间（已注释）
            nn.Tanh()  # 使用Tanh激活函数，将输出映射到[-1, 1]
        )

    def forward(self, x):
        # 前向传播函数
        # print('dec')  # 调试用，打印“dec”
        # print('input ', x.size())  # 可打印输入潜在向量尺寸
        x = self.fc(x)  # 通过全连接层扩展潜在向量
        x = x.view(-1, self.dim_h * 8, 7, 7)  
        # 将全连接层输出重塑为形状为[batch, dim_h*8, 7, 7]的特征图，为反卷积做准备
        x = self.deconv(x)  # 将特征图通过反卷积层还原成图像
        return x  # 返回重构后的图像

##############################################################################
# 定义辅助函数，用于生成SMOTE图像和获取特定类别数据

def biased_get_class1(c):
    # 根据给定类别c，从全局变量dec_x和dec_y中筛选出该类别的样本
    xbeg = dec_x[dec_y == c]  # 选择图像中标签等于c的部分
    ybeg = dec_y[dec_y == c]  # 选择标签中等于c的部分
    return xbeg, ybeg  # 返回该类别的图像和标签
    # return xclass, yclass  # 备用返回方式（已注释）

def G_SM1(X, y, n_to_sample, cl):
    # 根据SMOTE思想生成合成样本
    # fitting the model
    n_neigh = 5 + 1  # 设置最近邻数量为6（包含样本自身）
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)  # 初始化最近邻对象，使用单线程
    nn.fit(X)  # 用数据X拟合最近邻模型
    dist, ind = nn.kneighbors(X)  # 获取每个样本的邻居索引和距离

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    # 从所有样本中随机选择n_to_sample个作为基样本
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)
    # 对每个基样本随机选择一个邻居（排除自身索引0）

    X_base = X[base_indices]  # 获取基样本数据
    X_neighbor = X[ind[base_indices, neighbor_indices]]  # 获取对应邻居数据

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                   X_neighbor - X_base)
    # 在基样本与邻居之间做随机线性插值，生成新的合成样本

    # use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl] * n_to_sample
    # 返回生成的样本及其标签列表（所有生成样本标签均为cl）

#############################################################################
np.printoptions(precision=5, suppress=True)
# 设置NumPy打印选项：保留5位小数且不使用科学计数法（只影响打印输出，不影响计算）

# 定义训练图像和标签文件的路径
dtrnimg = '.../0_trn_img.txt'  # 图像文件的路径（或目录路径，具体依据实际情况）
dtrnlab = '.../0_trn_lab.txt'  # 标签文件的路径（或目录路径）

ids = os.listdir(dtrnimg)  
# 列出dtrnimg目录下所有文件的文件名
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
# 将目录路径与文件名组合成完整路径列表
print(idtri_f)  # 打印图像文件的完整路径列表

ids = os.listdir(dtrnlab)
# 列出dtrnlab目录下所有文件的文件名
idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]
# 将标签文件的目录与文件名组合成完整路径列表
print(idtrl_f)  # 打印标签文件的完整路径列表

# 定义存储模型的路径
modpth = '.../MNIST/models/crs5/'

encf = []  # 用于存放编码器模型文件的路径
decf = []  # 用于存放解码器模型文件的路径
for p in range(5):
    enc = modpth + '/' + str(p) + '/bst_enc.pth'  # 构造第p折交叉验证中最佳编码器模型的路径
    dec = modpth + '/' + str(p) + '/bst_dec.pth'  # 构造第p折中最佳解码器模型的路径
    encf.append(enc)  # 将编码器路径添加到列表中
    decf.append(dec)  # 将解码器路径添加到列表中
    # 可打印路径用于调试（已注释）
    # print(enc)
    # print(dec)
    # print()

# 对5折数据分别进行处理
for m in range(5):
    print(m)  # 打印当前折数
    trnimgfile = idtri_f[m]  # 获取当前折的训练图像文件路径
    trnlabfile = idtrl_f[m]  # 获取当前折的训练标签文件路径
    print(trnimgfile)  # 打印训练图像文件路径
    print(trnlabfile)  # 打印训练标签文件路径
    dec_x = np.loadtxt(trnimgfile)  # 从文本文件中加载训练图像数据（NumPy数组）
    dec_y = np.loadtxt(trnlabfile)  # 从文本文件中加载训练标签数据

    print('train imgs before reshape ', dec_x.shape)  
    # 打印加载的图像数据形状，例如(44993, 3072)或(45500, 3072)
    print('train labels ', dec_y.shape)  
    # 打印标签数据的形状，例如(44993,)或(45500,)

    dec_x = dec_x.reshape(dec_x.shape[0], 1, 28, 28)
    # 将图像数据重塑为[样本数, 通道数, 高, 宽]，这里假定图像大小为28x28，通道数为1

    print('decy ', dec_y.shape)  # 再次打印标签形状
    print(collections.Counter(dec_y))  # 统计并打印每个标签的样本数量分布
    
    print('train imgs after reshape ', dec_x.shape)  
    # 打印重塑后的图像数据形状，例如(45000, 1, 28, 28)

    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')  
    # 定义数字类别0-9的元组

    # 生成一些样本（平衡数据）：
    train_on_gpu = torch.cuda.is_available()  # 检查是否有可用GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 指定使用的设备

    path_enc = encf[m]  # 获取当前折的编码器模型文件路径
    path_dec = decf[m]  # 获取当前折的解码器模型文件路径

    encoder = Encoder(args)  # 初始化编码器模型
    encoder.load_state_dict(torch.load(path_enc), strict=False)  
    # 从指定路径加载编码器模型参数；strict=False表示允许部分匹配
    encoder = encoder.to(device)  # 将编码器模型移动到指定设备上

    decoder = Decoder(args)  # 初始化解码器模型
    decoder.load_state_dict(torch.load(path_dec), strict=False)  
    # 从指定路径加载解码器模型参数
    decoder = decoder.to(device)  # 将解码器模型移动到设备上

    encoder.eval()  # 将编码器设置为评估模式（关闭dropout、batchnorm更新等）
    decoder.eval()  # 将解码器设置为评估模式

    # 定义各类别的目标样本数（imbalanced样本数），用于生成合成样本
    imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    resx = []  # 用于存放生成的合成图像（重构后的图像）
    resy = []  # 用于存放生成的样本对应的标签

    for i in range(1, 10):
        # 遍历类别1到9（类别0通常为多数类，不做SMOTE扩充）
        xclass, yclass = biased_get_class1(i)  
        # 获取当前类别i的图像和标签
        print(xclass.shape)  # 打印该类别图像数据形状，例如(500, 1, 28, 28)
        print(yclass[0])  # 打印标签中第一个元素，确认标签内容

        # 将该类别图像数据转换为Tensor，并送入设备
        xclass = torch.Tensor(xclass)
        xclass = xclass.to(device)
        xclass = encoder(xclass)  
        # 通过编码器将图像转换到潜在空间，得到形状类似[样本数, n_z]的张量
        print(xclass.shape)  # 打印潜在表示的形状，例如(torch.Size([500, 300]))
            
        xclass = xclass.detach().cpu().numpy()  
        # 将潜在表示从Tensor转换为NumPy数组，方便后续SMOTE处理
        n = imbal[0] - imbal[i]  
        # 计算当前类别需要生成的样本数（目标多数类样本数减去当前类别样本数）
        xsamp, ysamp = G_SM1(xclass, yclass, n, i)  
        # 使用SMOTE函数生成新的潜在向量样本和对应标签
        print(xsamp.shape)  # 打印生成的样本形状，例如(4500, 300)
        print(len(ysamp))  # 打印生成的样本标签数量，例如4500
        ysamp = np.array(ysamp)  
        # 将标签列表转换为NumPy数组
        print(ysamp.shape)  # 打印标签数组形状，例如(4500,)
    
        """to generate samples for resnet"""
        xsamp = torch.Tensor(xsamp)  
        # 将生成的潜在向量样本转换为Tensor
        xsamp = xsamp.to(device)  
        # 将样本移到设备上
        # xsamp = xsamp.view(xsamp.size()[0], xsamp.size()[1], 1, 1)
        # 上面代码（已注释）用于将向量调整形状为[batch, n_z, 1, 1]，如果需要可启用
        # print(xsamp.size())  # 调试用，打印转换后张量的形状
        ximg = decoder(xsamp)  
        # 通过解码器将潜在向量转换为图像

        ximn = ximg.detach().cpu().numpy()  
        # 将解码器输出的图像从Tensor转换为NumPy数组
        print(ximn.shape)  # 打印生成图像的形状，例如(4500, 1, 28, 28)
        # ximn = np.expand_dims(ximn, axis=1)  # 可选扩展维度（已注释）
        print(ximn.shape)  # 再次打印图像数组形状，确认不变
        resx.append(ximn)  # 将生成的图像添加到列表resx中
        resy.append(ysamp)  # 将生成的标签添加到列表resy中
    
    resx1 = np.vstack(resx)  
    # 将列表中所有生成的图像在第一个维度上堆叠，形成一个大数组
    resy1 = np.hstack(resy)  
    # 将所有生成的标签在水平方向上堆叠成一维数组
    # print(resx1.shape)  # 可打印生成图像总形状，例如(34720, 1, 28, 28)
    print(resx1.shape)  # 打印最终生成图像数组的形状
    print(resy1.shape)  # 打印最终生成标签数组的形状

    resx1 = resx1.reshape(resx1.shape[0], -1)  
    # 将生成的图像数组重塑为二维，每行代表一张图（展平成向量）
    print(resx1.shape)  # 打印重塑后数组的形状，例如(34720, 3072)
    
    dec_x1 = dec_x.reshape(dec_x.shape[0], -1)  
    # 同样将原始训练图像数据展平成二维
    print('decx1 ', dec_x1.shape)  # 打印原始图像展平后的形状
    combx = np.vstack((resx1, dec_x1))  
    # 将生成的图像与原始图像在样本维度上堆叠，构成新的训练集图像数据
    comby = np.hstack((resy1, dec_y))  
    # 将生成的标签与原始标签在水平方向上拼接，构成新的训练集标签数据

    print(combx.shape)  # 打印合并后图像数据的形状，例如(45000, 3072)
    print(comby.shape)  # 打印合并后标签数据的形状，例如(45000,)

    ifile = '.../MNIST/trn_img_f/' + str(m) + '_trn_img.txt'  
    # 构造保存新训练图像数据的文件路径，文件名中包含当前折号
    np.savetxt(ifile, combx)  
    # 将合并后的图像数据保存为文本文件

    lfile = '.../MNIST/trn_lab_f/' + str(m) + '_trn_lab.txt'  
    # 构造保存新训练标签数据的文件路径
    np.savetxt(lfile, comby)  
    # 将合并后的标签数据保存为文本文件
    print()  # 打印空行，用于分隔不同折的输出

t1 = time.time()  # 记录整个程序结束时的时间
print('final time(min): {:.2f}'.format((t1 - t0) / 60))  
# 计算并打印整个程序运行的时间（单位：分钟）
