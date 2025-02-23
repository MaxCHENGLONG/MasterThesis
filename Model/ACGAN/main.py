from __future__ import print_function  # 引入未来版本的打印函数，确保兼容 Python 3 语法
import argparse  # 导入 argparse 模块，用于解析命令行参数
import os  # 导入 os 模块，用于文件和目录操作
import numpy as np  # 导入 numpy 模块并简写为 np，用于数值计算
import random  # 导入 random 模块，用于生成随机数
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块，并简写为 nn
import torch.nn.parallel  # 导入 PyTorch 的并行计算模块
import torch.backends.cudnn as cudnn  # 导入 cuDNN 后端，用于加速卷积计算
import torch.optim as optim  # 导入 PyTorch 优化器模块，并简写为 optim
import torch.utils.data  # 导入 PyTorch 数据加载工具模块
import torchvision.datasets as dset  # 导入 torchvision 数据集模块，并简写为 dset
import torchvision.transforms as transforms  # 导入 torchvision 数据预处理模块，并简写为 transforms
import torchvision.utils as vutils  # 导入 torchvision 工具模块，用于处理图像数据，并简写为 vutils
from torch.autograd import Variable  # 从 torch.autograd 模块中导入 Variable 类（用于封装张量，旧版本 PyTorch 使用）
from utils import weights_init, compute_acc  # 从 utils 模块导入自定义权重初始化和计算准确率的函数
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10  # 从 network 模块导入生成器和判别器网络
from folder import ImageFolder  # 从 folder 模块导入 ImageFolder 类，用于加载自定义图像数据
from Get_datasets import get_datasets # 从 Get_datasets 模块导入 get_datasets 函数，用于加载 MNIST-3/4 数据集
parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')  # 添加必需参数：数据集类型（cifar10 或 imagenet）
parser.add_argument('--dataroot', required=True, help='path to dataset')  # 添加必需参数：数据集路径
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)  # 添加参数：数据加载线程数，默认值为2
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')  # 添加参数：批次大小，默认值为1
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')  # 添加参数：输入图像的尺寸（宽高），默认128
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')  # 添加参数：潜在向量 z 的维度，默认110
parser.add_argument('--ngf', type=int, default=64)  # 添加参数：生成器特征图的基数，默认64
parser.add_argument('--ndf', type=int, default=64)  # 添加参数：判别器特征图的基数，默认64
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')  # 添加参数：训练轮数（epoch 数），默认25
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')  # 添加参数：学习率，默认 0.0002
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')  # 添加参数：Adam 优化器 beta1 参数，默认 0.5
parser.add_argument('--cuda', action='store_true', help='enables cuda')  # 添加参数：是否启用 CUDA（GPU 加速）
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')  # 添加参数：使用的 GPU 数量，默认1
parser.add_argument('--netG', default='', help="path to netG (to continue training)")  # 添加参数：生成器模型文件路径（用于继续训练）
parser.add_argument('--netD', default='', help="path to netD (to continue training)")  # 添加参数：判别器模型文件路径（用于继续训练）
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')  # 添加参数：输出文件夹，用于保存图像和模型检查点，默认当前目录
parser.add_argument('--manualSeed', type=int, help='manual seed')  # 添加参数：手动设置随机种子
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')  # 添加参数：AC-GAN 的类别数，默认 10
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')  # 添加参数：指定 GPU 的 ID，默认 0

opt = parser.parse_args()  # 解析命令行参数
print(opt)  # 打印解析后的参数信息
X_train = []
y_train = []
X_test = []
y_test = []
# 如果只使用一个 GPU，则指定使用的 GPU ID
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)  # 设置环境变量，只显示指定的 GPU

try:
    os.makedirs(opt.outf)  # 尝试创建输出文件夹
except OSError:
    pass  # 如果输出文件夹已存在，则忽略异常

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)  # 如果未手动设置随机种子，则随机生成一个
print("Random Seed: ", opt.manualSeed)  # 打印随机种子
random.seed(opt.manualSeed)  # 设置 Python 内置随机数种子
torch.manual_seed(opt.manualSeed)  # 设置 PyTorch 随机数种子
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)  # 如果使用 CUDA，则设置所有 GPU 的随机数种子

cudnn.benchmark = True  # 启用 cuDNN benchmark 模式以选择最优算法，加快训练

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")  # 如果有 CUDA 设备但未启用 CUDA，则发出警告

# 数据集加载部分
if opt.dataset == 'imagenet':
    # 如果选择的数据集是 imagenet，则使用自定义 ImageFolder 加载数据
    dataset = ImageFolder(
        root=opt.dataroot,  # 数据集根目录
        transform=transforms.Compose([  # 定义数据预处理操作
            transforms.Scale(opt.imageSize),  # 缩放图像到指定尺寸
            transforms.CenterCrop(opt.imageSize),  # 从中心裁剪图像到指定尺寸
            transforms.ToTensor(),  # 将图像转换为 Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 对图像进行归一化处理，使像素值归一化到 [-1,1]
        ]),
        classes_idx=(10, 20)  # 指定加载的类别索引范围
    )
elif opt.dataset == 'cifar10':
    # 如果选择的数据集是 cifar10，则使用 torchvision.datasets 加载数据
    dataset = dset.CIFAR10(
        root=opt.dataroot, download=True,  # 数据集存储路径，并自动下载数据（如果不存在）
        transform=transforms.Compose([  # 定义数据预处理操作
            transforms.Scale(opt.imageSize),  # 缩放图像到指定尺寸
            transforms.ToTensor(),  # 将图像转换为 Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 对图像进行归一化处理
        ]))
elif opt.dataset == 'mnist34':
    X_train, y_train, X_test, y_test = get_datasets(dataname="mnist34", fraction = 0.005)
else:
    raise NotImplementedError("No such dataset {}".format(opt.dataset))  # 如果数据集类型不支持，则抛出异常

assert dataset  # 确保数据集加载成功
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))  # 创建数据加载器，设置批次大小、打乱数据和工作线程数

# 设置一些超参数
ngpu = int(opt.ngpu)  # 获取 GPU 数量
nz = int(opt.nz)  # 获取潜在向量 z 的维度
ngf = int(opt.ngf)  # 获取生成器特征图基数
ndf = int(opt.ndf)  # 获取判别器特征图基数
num_classes = int(opt.num_classes)  # 获取 AC-GAN 的类别数
nc = 3  # 定义输入图像的通道数（3 表示 RGB 图像）

# 定义生成器网络并初始化权重
if opt.dataset == 'imagenet':
    netG = _netG(ngpu, nz)  # 如果数据集为 imagenet，使用 _netG 生成器
else:
    netG = _netG_CIFAR10(ngpu, nz)  # 如果数据集为 cifar10，使用 _netG_CIFAR10 生成器
netG.apply(weights_init)  # 对生成器网络中的所有模块应用自定义权重初始化
if opt.netG != '':  # 如果提供了预训练生成器模型路径，则加载模型参数
    netG.load_state_dict(torch.load(opt.netG))
print(netG)  # 打印生成器网络结构

# 定义判别器网络并初始化权重
if opt.dataset == 'imagenet':
    netD = _netD(ngpu, num_classes)  # 如果数据集为 imagenet，使用 _netD 判别器
else:
    netD = _netD_CIFAR10(ngpu, num_classes)  # 如果数据集为 cifar10，使用 _netD_CIFAR10 判别器
netD.apply(weights_init)  # 对判别器网络中的所有模块应用自定义权重初始化
if opt.netD != '':  # 如果提供了预训练判别器模型路径，则加载模型参数
    netD.load_state_dict(torch.load(opt.netD))
print(netD)  # 打印判别器网络结构

# 定义损失函数
dis_criterion = nn.BCELoss()  # 使用二元交叉熵损失函数计算判别器的真伪损失
aux_criterion = nn.NLLLoss()  # 使用负对数似然损失函数计算辅助分类器的损失

# 定义张量占位符
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 创建存储输入图像的张量，占位尺寸为 [batchSize, 3, imageSize, imageSize]
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)  # 创建存储生成器噪声的张量，占位尺寸为 [batchSize, nz, 1, 1]
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)  # 创建评估用噪声张量，并用标准正态分布初始化
dis_label = torch.FloatTensor(opt.batchSize)  # 创建存储判别器标签的张量，占位尺寸为 [batchSize]
aux_label = torch.LongTensor(opt.batchSize)  # 创建存储辅助分类标签的张量，占位尺寸为 [batchSize]
real_label = 1  # 定义真实图像标签值为 1
fake_label = 0  # 定义生成图像标签值为 0

# 如果使用 CUDA，则将网络和张量移动到 GPU 上
if opt.cuda:
    netD.cuda()  # 将判别器移动到 GPU
    netG.cuda()  # 将生成器移动到 GPU
    dis_criterion.cuda()  # 将判别器损失函数移动到 GPU
    aux_criterion.cuda()  # 将辅助分类损失函数移动到 GPU
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()  # 将输入和标签张量移动到 GPU
    noise, eval_noise = noise.cuda(), eval_noise.cuda()  # 将噪声张量移动到 GPU

# 将张量封装成 Variable（旧版 PyTorch 用法，现在已合并到 Tensor 中）
input = Variable(input)  # 封装输入图像张量
noise = Variable(noise)  # 封装生成器噪声张量
eval_noise = Variable(eval_noise)  # 封装评估用噪声张量
dis_label = Variable(dis_label)  # 封装判别器标签张量
aux_label = Variable(aux_label)  # 封装辅助分类标签张量

# 生成评估用的噪声和标签
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))  # 使用正态分布生成评估噪声，尺寸为 [batchSize, nz]
eval_label = np.random.randint(0, num_classes, opt.batchSize)  # 随机生成评估标签，取值范围在 0 到 num_classes-1
eval_onehot = np.zeros((opt.batchSize, num_classes))  # 创建评估用的 one-hot 编码矩阵，尺寸为 [batchSize, num_classes]
eval_onehot[np.arange(opt.batchSize), eval_label] = 1  # 根据生成的标签设置 one-hot 编码
eval_noise_[np.arange(opt.batchSize), :num_classes] = eval_onehot[np.arange(opt.batchSize)]  # 将 one-hot 编码嵌入评估噪声的前 num_classes 维
eval_noise_ = (torch.from_numpy(eval_noise_))  # 将 numpy 数组转换为 PyTorch 张量
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))  # 将生成的评估噪声复制到 eval_noise 变量中，并调整形状

# 设置优化器
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 使用 Adam 优化器优化判别器参数
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 使用 Adam 优化器优化生成器参数

avg_loss_D = 0.0  # 初始化判别器平均损失为 0
avg_loss_G = 0.0  # 初始化生成器平均损失为 0
avg_loss_A = 0.0  # 初始化辅助分类器平均准确率为 0

for epoch in range(opt.niter):  # 循环遍历每个 epoch（训练轮数）
    for i, data in enumerate(dataloader, 0):  # 遍历数据加载器中的每个批次
        ############################
        # (1) 更新判别器网络：最大化 log(D(x)) + log(1 - D(G(z)))
        ############################
        # 用真实数据训练判别器
        netD.zero_grad()  # 清空判别器的梯度
        real_cpu, label = data  # 获取当前批次的真实图像和对应标签
        batch_size = real_cpu.size(0)  # 获取当前批次的样本数
        if opt.cuda:
            real_cpu = real_cpu.cuda()  # 如果使用 CUDA，将真实图像移动到 GPU
        input.data.resize_as_(real_cpu).copy_(real_cpu)  # 调整 input 张量大小以匹配真实图像，并复制图像数据
        dis_label.data.resize_(batch_size).fill_(real_label)  # 将判别器标签调整为当前批次大小，并填充为真实标签
        aux_label.data.resize_(batch_size).copy_(label)  # 将辅助分类标签调整为当前批次大小，并复制真实标签
        dis_output, aux_output = netD(input)  # 将真实图像输入判别器，获得真伪输出和辅助分类输出

        dis_errD_real = dis_criterion(dis_output, dis_label)  # 计算真实图像的判别器损失
        aux_errD_real = aux_criterion(aux_output, aux_label)  # 计算真实图像的辅助分类损失
        errD_real = dis_errD_real + aux_errD_real  # 计算真实图像总损失（判别器损失 + 辅助分类损失）
        errD_real.backward()  # 对真实图像损失进行反向传播，计算梯度
        D_x = dis_output.data.mean()  # 计算真实图像输出的平均值，用于监控

        # 计算当前批次的分类准确率
        accuracy = compute_acc(aux_output, aux_label)  # 调用 compute_acc 函数计算辅助分类器的准确率

        # 用生成的假数据训练判别器
        noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)  # 生成随机噪声，并调整其形状为 [batchSize, nz, 1, 1]
        label = np.random.randint(0, num_classes, batch_size)  # 随机生成当前批次的标签，用作条件输入
        noise_ = np.random.normal(0, 1, (batch_size, nz))  # 生成噪声数组，尺寸为 [batchSize, nz]
        class_onehot = np.zeros((batch_size, num_classes))  # 创建一个全零矩阵，用于生成 one-hot 编码
        class_onehot[np.arange(batch_size), label] = 1  # 根据随机标签生成 one-hot 编码
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]  # 将 one-hot 编码嵌入噪声的前 num_classes 维
        noise_ = (torch.from_numpy(noise_))  # 将 numpy 数组转换为 PyTorch 张量
        noise.data.copy_(noise_.view(batch_size, nz, 1, 1))  # 将处理好的噪声复制到 noise 变量中，并调整形状
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))  # 将生成的标签转换为张量并复制到辅助标签变量中

        fake = netG(noise)  # 使用生成器生成假图像
        dis_label.data.fill_(fake_label)  # 将判别器标签全部设置为虚假标签
        dis_output, aux_output = netD(fake.detach())  # 将生成的假图像输入判别器，detach() 防止梯度传回生成器
        dis_errD_fake = dis_criterion(dis_output, dis_label)  # 计算假图像的判别器损失
        aux_errD_fake = aux_criterion(aux_output, aux_label)  # 计算假图像的辅助分类损失
        errD_fake = dis_errD_fake + aux_errD_fake  # 计算假图像总损失（判别器损失 + 辅助分类损失）
        errD_fake.backward()  # 对假图像损失进行反向传播，计算梯度
        D_G_z1 = dis_output.data.mean()  # 计算生成假图像的判别器输出平均值，用于监控
        errD = errD_real + errD_fake  # 计算判别器总损失（真实 + 假数据的损失之和）
        optimizerD.step()  # 更新判别器参数

        ############################
        # (2) 更新生成器网络：最大化 log(D(G(z)))
        ############################
        netG.zero_grad()  # 清空生成器的梯度
        dis_label.data.fill_(real_label)  # 将判别器标签设置为真实标签（生成器目标是欺骗判别器）
        dis_output, aux_output = netD(fake)  # 将生成的假图像输入判别器，获得输出
        dis_errG = dis_criterion(dis_output, dis_label)  # 计算生成器的判别器损失
        aux_errG = aux_criterion(aux_output, aux_label)  # 计算生成器的辅助分类损失
        errG = dis_errG + aux_errG  # 计算生成器总损失（判别器损失 + 辅助分类损失）
        errG.backward()  # 对生成器损失进行反向传播，计算梯度
        D_G_z2 = dis_output.data.mean()  # 计算生成图像在判别器中的输出平均值，用于监控
        optimizerG.step()  # 更新生成器参数

        # 计算当前的平均损失和平均准确率
        curr_iter = epoch * len(dataloader) + i  # 计算当前迭代的总步数
        all_loss_G = avg_loss_G * curr_iter  # 累加之前的生成器总损失
        all_loss_D = avg_loss_D * curr_iter  # 累加之前的判别器总损失
        all_loss_A = avg_loss_A * curr_iter  # 累加之前的辅助分类准确率总和
        all_loss_G += errG.data[0]  # 加入当前生成器损失
        all_loss_D += errD.data[0]  # 加入当前判别器损失
        all_loss_A += accuracy  # 加入当前准确率
        avg_loss_G = all_loss_G / (curr_iter + 1)  # 计算新的生成器平均损失
        avg_loss_D = all_loss_D / (curr_iter + 1)  # 计算新的判别器平均损失
        avg_loss_A = all_loss_A / (curr_iter + 1)  # 计算新的辅助分类平均准确率

        # 打印当前训练状态，包括损失、判别器输出和准确率
        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], avg_loss_D, errG.data[0], avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
        if i % 100 == 0:  # 每100个批次执行一次保存操作
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf)  # 保存当前批次的真实图像样本到指定文件夹
            print('Label for eval = {}'.format(eval_label))  # 打印评估时使用的标签
            fake = netG(eval_noise)  # 用评估噪声生成假图像
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)  # 保存生成器生成的假图像样本，文件名中包含当前 epoch
            )

    # 每个 epoch 结束后，保存当前的模型检查点
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))  # 保存生成器参数
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))  # 保存判别器参数