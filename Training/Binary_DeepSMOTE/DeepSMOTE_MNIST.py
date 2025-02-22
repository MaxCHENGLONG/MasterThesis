import collections  # 导入collections模块，用于统计和操作容器数据，如Counter
import torch  # 导入PyTorch库，用于深度学习任务
import torch.nn as nn  # 从torch中导入神经网络模块，简化模型构建
from torch.utils.data import TensorDataset  # 导入TensorDataset，用于将Tensor数据打包成数据集
import numpy as np  # 导入NumPy库，用于高效的数值计算和数组操作
from sklearn.neighbors import NearestNeighbors  # 导入最近邻算法，用于在SMOTE中寻找相邻样本
import time  # 导入time模块，用于计时
import os  # 导入os模块，用于文件和目录操作

print(torch.version.cuda)  # 打印当前PyTorch使用的CUDA版本，例如显示"10.1"
t3 = time.time()  # 记录程序开始时的时间，用于后续计算总运行时间

##############################################################################
"""args for AE"""
# 以下部分设置自动编码器（AE）的相关参数
args = {}  # 创建一个空字典，用于存放模型和训练的参数
args['dim_h'] = 64         # 设置隐藏层通道数的基础因子，后续卷积层的通道数会成倍增加
args['n_channel'] = 1  #3    # 输入数据的通道数，1表示灰度图（3则为彩色图）；这里选用灰度图
args['n_z'] = 300 #600     # 潜在空间（编码空间）的维度数，决定编码器输出特征向量的大小
args['sigma'] = 1.0        # 潜在空间中使用的方差参数，可用于正则化
args['lambda'] = 0.01      # 判别器损失的权重超参数（如在对抗训练中使用）
args['lr'] = 0.0002        # Adam优化器的学习率，决定参数更新的步长
args['epochs'] = 50       # 训练过程中遍历数据集的轮数
args['batch_size'] = 64   # 每个训练批次的样本数量
args['save'] = True        # 如果为True，则在每个训练轮结束时保存模型权重
args['train'] = True       # 若为True则进行训练，否则加载已保存的模型进行测试
args['dataset'] = 'mnist'  #'fmnist' # 指定使用的数据集，这里选择MNIST数据集

##############################################################################
# 定义编码器模型
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()  # 调用父类构造函数
        self.n_channel = args['n_channel']  # 获取输入数据的通道数
        self.dim_h = args['dim_h']          # 获取隐藏层基本通道数
        self.n_z = args['n_z']              # 获取潜在空间的维度数
        
        # 使用卷积层提取图像特征
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),  
            # 第一层卷积：输入通道数为n_channel，输出为dim_h，卷积核大小4，步幅2，填充1，无偏置
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),  # 使用LeakyReLU激活函数，负半部斜率设为0.2
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),  
            # 第二层卷积：通道数翻倍到dim_h*2
            nn.BatchNorm2d(self.dim_h * 2),  # 对第二层卷积输出进行批归一化
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),  
            # 第三层卷积：通道数增加到dim_h*4
            nn.BatchNorm2d(self.dim_h * 4),  # 批归一化
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),  # 激活函数
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),  
            # 第四层卷积：通道数增加到dim_h*8
            
            #3d and 32 by 32
            #nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),  # 备用卷积层配置
            nn.BatchNorm2d(self.dim_h * 8),  # 对第四层卷积输出进行批归一化
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True)  # 激活函数
            # 注释中还有其他可能的卷积配置，这里使用的是标准配置
        )
        # 全连接层：将卷积层输出映射到潜在空间
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)  
        # 这里计算dim_h * (2**3)相当于dim_h*8，假设卷积层最后输出特征数为dim_h*8

    def forward(self, x):
        # 前向传播函数
        # print('enc')  # 调试打印，可查看编码器被调用
        # print('input ', x.size())  # 打印输入尺寸，例如torch.Size([batch_size, channel, H, W])
        x = self.conv(x)  # 将输入图像通过卷积层提取特征
        x = x.squeeze()   # 去除多余的尺寸（例如将[batch_size, 1, N]变为[batch_size, N]）
        # print('aft squeeze ', x.size())  # 调试打印压缩后的尺寸
        x = self.fc(x)    # 通过全连接层映射到潜在空间维度
        # print('out ', x.size())  # 打印最终输出尺寸，应为[batch_size, n_z]
        return x  # 返回编码后的潜在表示

# 定义解码器模型
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()  # 调用父类构造函数
        self.n_channel = args['n_channel']  # 获取输入通道数（用于输出重构图像）
        self.dim_h = args['dim_h']          # 获取隐藏层基本通道数
        self.n_z = args['n_z']              # 获取潜在空间的维度数

        # 全连接层：将潜在向量映射到足够重构卷积特征图的尺寸
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),  # 将潜在向量转换为高维特征，尺寸为[batch_size, dim_h*8*7*7]
            nn.ReLU()  # 使用ReLU激活函数
        )

        # 反卷积层（转置卷积）：将全连接层的输出转换为图像
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),  
            # 第一层反卷积：将通道数从dim_h*8降到dim_h*4，卷积核大小4
            nn.BatchNorm2d(self.dim_h * 4),  # 批归一化
            nn.ReLU(True),  # 激活函数
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),  
            # 第二层反卷积：将通道数从dim_h*4降到dim_h*2
            nn.BatchNorm2d(self.dim_h * 2),  # 批归一化
            nn.ReLU(True),  # 激活函数
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),  
            # 第三层反卷积：将通道数降为1，同时上采样（步幅为2），恢复到原图大小
            # nn.Sigmoid())  # 也可用Sigmoid激活函数使输出在[0,1]之间
            nn.Tanh()  # 这里使用Tanh激活函数，将输出映射到[-1,1]
        )

    def forward(self, x):
        # 前向传播函数
        # print('dec')  # 调试打印，查看解码器调用
        # print('input ', x.size())  # 打印输入潜在向量的尺寸
        x = self.fc(x)  # 通过全连接层处理潜在向量
        x = x.view(-1, self.dim_h * 8, 7, 7)  
        # 将全连接层输出重塑为特征图，尺寸为[batch_size, dim_h*8, 7, 7]，为反卷积做准备
        x = self.deconv(x)  # 通过反卷积层重构出图像
        return x  # 返回重构图像

##############################################################################
"""set models, loss functions"""
# 以下函数用于控制模块参数是否参与梯度更新

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True  # 将该模块中所有参数设置为参与梯度计算（训练状态）

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False  # 将该模块中所有参数冻结，不进行梯度更新

##############################################################################
"""functions to create SMOTE images"""
# 以下函数用于生成SMOTE（Synthetic Minority Over-sampling Technique）图像

def biased_get_class(c):
    xbeg = dec_x[dec_y == c]  # 从全局训练图像dec_x中选择标签等于c的所有样本
    ybeg = dec_y[dec_y == c]  # 从全局标签dec_y中选择标签等于c的所有样本
    return xbeg, ybeg  # 返回该类别的图像和标签
    # return xclass, yclass  # 注释掉的另一种返回方式

def G_SM(X, y, n_to_sample, cl):
    # 此函数根据SMOTE思想生成新的样本
    # determining the number of samples to generate
    # n_to_sample = 10  # 示例：生成10个样本

    # fitting the model
    n_neigh = 5 + 1  # 设置最近邻数为6（包括自身），实际选取邻居时会跳过自身
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)  # 初始化最近邻模型，n_jobs=1指定使用单线程
    nn.fit(X)  # 使用数据X拟合最近邻模型
    dist, ind = nn.kneighbors(X)  # 对每个样本找到最近邻样本的距离和索引

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)  
    # 随机选择n_to_sample个样本作为基样本
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)  
    # 为每个基样本随机选择一个邻居索引（范围从1到n_neigh-1，排除自身索引0）

    X_base = X[base_indices]  # 选取基样本
    X_neighbor = X[ind[base_indices, neighbor_indices]]  # 根据邻居索引选取对应的邻居样本

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
                                   X_neighbor - X_base)  
    # 在基样本和邻居样本之间随机插值，生成新的合成样本

    # use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl] * n_to_sample  
    # 返回生成的样本以及对应的标签列表（所有样本标签均为cl）

# xsamp, ysamp = SM(xclass, yclass)  # 注释：可用此行测试SMOTE函数

###############################################################################
# NOTE: 以下注释说明了数据集的准备要求
# NOTE: Download the training ('.../0_trn_img.txt') and label files 
# ('.../0_trn_lab.txt').  Place the files in directories (e.g., ../MNIST/trn_img/
# and /MNIST/trn_lab/).  Originally, when the code was written, it was for 5 fold
# cross validation and hence there were 5 files in each of the 
# directories.  Here, for illustration, we use only 1 training and 1 label
# file (e.g., '.../0_trn_img.txt' and '.../0_trn_lab.txt').

dtrnimg = '.../MNIST/trn_img/'  # 指定存放训练图像的目录
dtrnlab = '.../MNIST/trn_lab/'  # 指定存放训练标签的目录

ids = os.listdir(dtrnimg)  # 获取训练图像目录下所有文件的文件名列表
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]  
# 将图像文件名与目录路径拼接成完整路径
print(idtri_f)  # 打印训练图像文件的完整路径列表

ids = os.listdir(dtrnlab)  # 获取训练标签目录下所有文件的文件名列表
idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]  
# 将标签文件名与目录路径拼接成完整路径
print(idtrl_f)  # 打印训练标签文件的完整路径列表

# for i in range(5):
for i in range(len(ids)):  # 对每个数据文件进行处理（这里用于交叉验证，循环遍历每个fold）
    print()
    print(i)  # 打印当前处理的fold编号
    encoder = Encoder(args)  # 初始化编码器模型
    decoder = Decoder(args)  # 初始化解码器模型

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    # 检查是否有可用GPU，若有则使用'cuda'，否则使用'cpu'
    print(device)  # 打印当前使用的设备
    decoder = decoder.to(device)  # 将解码器模型移动到指定设备上
    encoder = encoder.to(device)  # 将编码器模型移动到指定设备上

    train_on_gpu = torch.cuda.is_available()  # 布尔变量，指示是否使用GPU训练

    # decoder loss function
    criterion = nn.MSELoss()  # 定义均方误差（MSE）损失函数，用于重构损失计算
    criterion = criterion.to(device)  # 将损失函数迁移到指定设备

    trnimgfile = idtri_f[i]  # 获取当前fold的训练图像文件路径
    trnlabfile = idtrl_f[i]  # 获取当前fold的训练标签文件路径
    
    print(trnimgfile)  # 打印当前训练图像文件路径
    print(trnlabfile)  # 打印当前训练标签文件路径
    dec_x = np.loadtxt(trnimgfile)  # 从文本文件中加载训练图像数据
    dec_y = np.loadtxt(trnlabfile)  # 从文本文件中加载训练标签数据

    print('train imgs before reshape ', dec_x.shape)  # 打印加载后的图像数据形状
    print('train labels ', dec_y.shape)  # 打印加载后的标签数据形状
    print(collections.Counter(dec_y))  # 统计并打印各类别标签的数量分布
    dec_x = dec_x.reshape(dec_x.shape[0], 1, 28, 28)   
    # 将图像数据重塑为[样本数, 通道数, 高, 宽]，这里假定图像大小为28x28，通道数为1
    print('train imgs after reshape ', dec_x.shape)  # 打印重塑后的图像数据形状

    batch_size = 64  # 定义训练时每个批次的样本数
    num_workers = 0   # 定义数据加载时使用的子进程数，0表示不使用多进程

    # torch.Tensor returns float so if want long then use torch.tensor
    tensor_x = torch.Tensor(dec_x)  # 将NumPy数组转换为FloatTensor（默认数据类型为float）
    tensor_y = torch.tensor(dec_y, dtype=torch.long)  # 将标签转换为LongTensor，因为标签通常为整数
    mnist_bal = TensorDataset(tensor_x, tensor_y)  
    # 使用TensorDataset将图像和标签封装成数据集，便于后续加载
    train_loader = torch.utils.data.DataLoader(mnist_bal, 
                                               batch_size=batch_size, shuffle=True, num_workers=num_workers)  
    # 利用DataLoader对数据集进行批量加载，并设置随机打乱

    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')  # 定义一个元组，表示数字类别0-9

    best_loss = np.inf  # 初始化最佳损失为无穷大，用于记录训练过程中的最低损失

    t0 = time.time()  # 记录当前fold训练开始的时间
    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])  
        # 使用Adam优化器对编码器参数进行优化
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])  
        # 使用Adam优化器对解码器参数进行优化
    
        for epoch in range(args['epochs']):  # 训练循环：遍历每个epoch
            train_loss = 0.0  # 初始化累积总损失
            tmse_loss = 0.0   # 初始化累积均方误差（MSE）损失（图像重构部分）
            tdiscr_loss = 0.0 # 初始化累积判别器/对比损失（SMOTE对比部分）
            # train for one epoch -- set nets to train mode
            encoder.train()  # 将编码器设置为训练模式（启用dropout、batchnorm等）
            decoder.train()  # 将解码器设置为训练模式
        
            for images, labs in train_loader:  # 遍历DataLoader中的每个批次
                # zero gradients for each batch
                encoder.zero_grad()  # 清除编码器上一批次计算的梯度
                decoder.zero_grad()  # 清除解码器上一批次计算的梯度
                # print(images)
                images, labs = images.to(device), labs.to(device)  
                # 将当前批次的数据迁移到指定设备（GPU或CPU）
                # print('images ', images.size()) 
                labsn = labs.detach().cpu().numpy()  
                # 将标签从Tensor转换为NumPy数组（通常用于后续非Tensor操作）
                # print('labsn ', labsn.shape, labsn)
            
                # run images
                z_hat = encoder(images)  
                # 将图像通过编码器得到潜在表示z_hat
            
                x_hat = decoder(z_hat)  
                # 将潜在表示通过解码器重构图像x_hat（输出通过Tanh激活函数）
                # print('xhat ', x_hat.size())
                # print(x_hat)
                mse = criterion(x_hat, images)  
                # 计算重构图像与原图像之间的均方误差（MSE）损失
                # print('mse ', mse)
                
                resx = []  # 用于存放后续生成样本（此处未实际使用）
                resy = []  # 用于存放生成样本对应的标签（此处未实际使用）
            
                tc = np.random.choice(10, 1)  
                # 随机选择一个类别（0-9中的一个），用于生成SMOTE样本
                # tc = 9  # 可选择固定类别，例如9
                xbeg = dec_x[dec_y == tc]  
                # 从全局训练图像中选取标签等于tc的所有图像
                ybeg = dec_y[dec_y == tc]  
                # 从全局标签中选取标签等于tc的所有标签
                xlen = len(xbeg)  # 计算选取到的样本数量
                nsamp = min(xlen, 100)  
                # 选取样本数，最多不超过100个（防止某类别样本过多）
                ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)  
                # 随机选取nsamp个样本的索引，不重复抽样
                xclass = xbeg[ind]  # 根据随机索引选取图像样本
                yclass = ybeg[ind]  # 根据随机索引选取标签样本
            
                xclen = len(xclass)  # 计算选取样本的数量
                # print('xclen ', xclen)
                xcminus = np.arange(1, xclen)  
                # 生成从1到xclen-1的索引数组，用于后续构造正样本对（对比学习）
                # print('minus ', xcminus.shape, xcminus)
                
                xcplus = np.append(xcminus, 0)  
                # 在索引数组后添加0，构成循环移位后的索引，用于构造一一对应的正样本
                # print('xcplus ', xcplus)
                xcnew = (xclass[[xcplus], :])  
                # 根据重新排列的索引从xclass中构造新的图像排列，用作正样本对
                # xcnew = np.squeeze(xcnew)
                xcnew = xcnew.reshape(xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4])  
                # 重塑xcnew的形状，使其与原图像尺寸一致
                # print('xcnew ', xcnew.shape)
            
                xcnew = torch.Tensor(xcnew)  
                # 将重新排列后的图像转换为Tensor
                xcnew = xcnew.to(device)  
                # 将Tensor迁移到指定设备上
            
                # encode xclass to feature space
                xclass = torch.Tensor(xclass)  
                # 将选取的原始图像样本转换为Tensor
                xclass = xclass.to(device)  # 迁移到设备
                xclass = encoder(xclass)  
                # 通过编码器获得xclass的潜在表示
                # print('xclass ', xclass.shape) 
            
                xclass = xclass.detach().cpu().numpy()  
                # 将编码后的潜在表示转换为NumPy数组，便于使用NumPy进行索引操作
                xc_enc = (xclass[[xcplus], :])  
                # 根据循环移位的索引重新排列潜在表示，构造正样本对
                xc_enc = np.squeeze(xc_enc)  # 去除冗余的维度
                # print('xc enc ', xc_enc.shape)
            
                xc_enc = torch.Tensor(xc_enc)  # 将重新排列后的潜在表示转换回Tensor
                xc_enc = xc_enc.to(device)  # 迁移到设备上
                
                ximg = decoder(xc_enc)  
                # 将重新排列后的潜在表示通过解码器生成新图像，用于对比学习损失计算
                mse2 = criterion(ximg, xcnew)  
                # 计算生成的新图像与原排列图像之间的均方误差损失（对比损失）
            
                comb_loss = mse2 + mse  
                # 将重构损失和对比损失相加，作为当前批次的总损失
                comb_loss.backward()  # 对总损失进行反向传播，计算梯度
            
                enc_optim.step()  # 更新编码器参数
                dec_optim.step()  # 更新解码器参数
            
                train_loss += comb_loss.item() * images.size(0)  
                # 累加本批次的总损失（乘以当前批次样本数以便后续求平均）
                tmse_loss += mse.item() * images.size(0)  
                # 累加本批次的重构损失
                tdiscr_loss += mse2.item() * images.size(0)  
                # 累加本批次的对比损失
            
            # print avg training statistics 
            train_loss = train_loss / len(train_loader)  
            # 计算每个epoch的平均总损失
            tmse_loss = tmse_loss / len(train_loader)  
            # 计算平均重构损失
            tdiscr_loss = tdiscr_loss / len(train_loader)  
            # 计算平均对比损失
            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                  train_loss, tmse_loss, tdiscr_loss))  
            # 打印当前epoch的损失信息

            # store the best encoder and decoder models
            # here, /crs5 is a reference to 5 way cross validation, but is not
            # necessary for illustration purposes
            if train_loss < best_loss:  
                # 如果当前epoch的平均损失低于历史最佳损失，则保存模型
                print('Saving..')
                path_enc = '.../MNIST/models/crs5/' + str(i) + '/bst_enc.pth'  
                # 构造保存最佳编码器模型的文件路径
                path_dec = '.../MNIST/models/crs5/' + str(i) + '/bst_dec.pth'  
                # 构造保存最佳解码器模型的文件路径
             
                torch.save(encoder.state_dict(), path_enc)  
                # 保存编码器当前状态字典（权重参数）
                torch.save(decoder.state_dict(), path_dec)  
                # 保存解码器当前状态字典
                
                best_loss = train_loss  # 更新历史最佳损失值
        
        # in addition, store the final model (may not be the best) for
        # informational purposes
        path_enc = '.../MNIST/models/crs5/' + str(i) + '/f_enc.pth'  
        # 构造保存最终编码器模型（可能不是最佳）的文件路径
        path_dec = '.../MNIST/models/crs5/' + str(i) + '/f_dec.pth'  
        # 构造保存最终解码器模型的文件路径
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)  # 保存最终编码器状态
        torch.save(decoder.state_dict(), path_dec)  # 保存最终解码器状态
        print()
              
    t1 = time.time()  # 记录当前fold训练结束时间
    print('total time(min): {:.2f}'.format((t1 - t0) / 60))  
    # 输出当前fold训练耗时（单位：分钟）
 
t4 = time.time()  # 记录整个程序结束时的时间
print('final time(min): {:.2f}'.format((t4 - t3) / 60))  
# 输出整个程序运行的总耗时（单位：分钟）
