import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入函数式接口
import numpy as np  # 导入 numpy 库
import os  # 导入 os 模块，用于文件操作
import re  # 导入正则表达式模块
import csv  # 导入 csv 模块
from collections import defaultdict  # 导入 defaultdict，用于记录训练历史
from PIL import Image  # 导入 PIL.Image，用于图像处理

# 假设存在一个保存图像数组的函数，类似于原代码中的 save_image_array
def save_image_array(img_array, filename):
    # 此处仅保存数组中的第一幅图像作为示例
    img = img_array[0][0]  # 取出第一幅图像
    # 将图像数值从[-1,1]映射到[0,255]
    img = (img.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)  # 转换为 numpy 数组并归一化
    im = Image.fromarray(img)  # 利用 PIL 构造图像对象
    im.save(filename)  # 保存图像到文件

# 定义 BalancingGAN 类
class BalancingGAN:
    # 构建生成器网络（使用全连接层 + 卷积上采样）
    def build_generator(self, latent_size, init_resolution=8):
        self.resolution = self.image_shape[1]         # 设置目标图像分辨率
        self.channels = self.image_shape[0]             # 设置目标图像通道数
        self.latent_size = latent_size                  # 潜在向量维度
        self.init_resolution = init_resolution          # 初始分辨率

        # 全连接部分：将潜在向量映射为形状为 (batch, 128*init_resolution*init_resolution) 的向量
        self.gen_fc = nn.Sequential(                   # 定义全连接层序列
            nn.Linear(latent_size, 1024, bias=False),    # 线性层：latent_size -> 1024
            nn.ReLU(inplace=True),                       # ReLU 激活函数
            nn.Linear(1024, 128 * init_resolution * init_resolution, bias=False),  # 线性层：1024 -> 128*init_resolution^2
            nn.ReLU(inplace=True)                        # ReLU 激活函数
        )

        # 卷积上采样部分：从初始特征图不断上采样到目标分辨率
        conv_layers = []                                # 存储卷积层的列表
        crt_res = init_resolution                       # 当前分辨率初始化为 init_resolution
        in_channels = 128                               # 初始特征图通道数为 128
        while crt_res < self.resolution:                # 当当前分辨率小于目标分辨率时
            conv_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 添加上采样层，2 倍上采样（最近邻插值）
            # 根据当前分辨率选择卷积层的输出通道数
            if crt_res < self.resolution // 2:
                out_channels = 256                     # 较低分辨率时使用 256 个滤波器
            else:
                out_channels = 128                     # 否则使用 128 个滤波器
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False))  # 卷积层（核大小 5，padding=2 保持尺寸）
            conv_layers.append(nn.ReLU(inplace=True))  # ReLU 激活函数
            in_channels = out_channels                 # 更新输入通道数
            crt_res *= 2                               # 分辨率翻倍
        # 最后一层卷积，将通道数转换为目标图像通道数，并用 Tanh 限制输出范围
        conv_layers.append(nn.Conv2d(in_channels, self.channels, kernel_size=3, padding=1, bias=False))  # 最后一层卷积（核大小 2，padding=1）
        conv_layers.append(nn.Tanh())                   # Tanh 激活函数，将输出映射到 [-1, 1]
        self.gen_conv = nn.Sequential(*conv_layers)     # 将卷积层组合成 Sequential 模块

    # 根据潜在向量生成图像（生成器前向传播）
    def generate_from_latent(self, latent):
        x = self.gen_fc(latent)                                         # 通过全连接层生成初始特征向量
        x = x.view(-1, 128, self.init_resolution, self.init_resolution)   # 重塑为 (batch, 128, init_resolution, init_resolution)
        img = self.gen_conv(x)                                          # 通过卷积上采样模块生成图像
        return img                                                    # 返回生成图像

    # 构建通用编码器（用于重构器和鉴别器）
    def build_common_encoder(self, min_latent_res=8):
        layers = []                                                   # 用于存储编码器各层
        layers.append(nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1))  # 卷积层：channels -> 32，步幅 2
        layers.append(nn.LeakyReLU(0.2, inplace=True))                # LeakyReLU 激活函数
        layers.append(nn.Dropout(0.3))                                # Dropout 防止过拟合
        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))  # 卷积层：32 -> 64
        layers.append(nn.LeakyReLU(0.2, inplace=True))                # LeakyReLU 激活函数
        layers.append(nn.Dropout(0.3))                                # Dropout 层
        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))  # 卷积层：64 -> 128，步幅 2
        layers.append(nn.LeakyReLU(0.2, inplace=True))                # LeakyReLU 激活函数
        layers.append(nn.Dropout(0.3))                                # Dropout 层
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))  # 卷积层：128 -> 256
        layers.append(nn.LeakyReLU(0.2, inplace=True))                # LeakyReLU 激活函数
        layers.append(nn.Dropout(0.3))                                # Dropout 层
        current_res = self.resolution // 4                            # 初始分辨率经过两次下采样后
        while current_res > min_latent_res:                           # 当当前分辨率大于最小潜在分辨率时
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1))  # 卷积层，下采样
            layers.append(nn.LeakyReLU(0.2, inplace=True))            # LeakyReLU 激活函数
            layers.append(nn.Dropout(0.3))                            # Dropout 层
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))  # 卷积层，不改变尺寸
            layers.append(nn.LeakyReLU(0.2, inplace=True))            # LeakyReLU 激活函数
            layers.append(nn.Dropout(0.3))                            # Dropout 层
            current_res = current_res // 2                            # 分辨率减半
        layers.append(nn.Flatten())                                   # 将特征展平为一维向量
        return nn.Sequential(*layers)                                 # 返回组合后的编码器

    # 构建重构器，将图像映射到潜在向量（编码器 + 全连接层）
    def build_reconstructor(self, latent_size, min_latent_res=8):
        self.encoder = self.build_common_encoder(min_latent_res)       # 使用通用编码器构建 encoder
        encoder_output_dim = self._get_encoder_output_dim()            # 获取编码器输出的展平维度
        self.reconstructor_fc = nn.Linear(encoder_output_dim, latent_size)  # 全连接层，将编码器输出映射到 latent_size

    # 辅助函数：计算编码器输出展平后的维度
    def _get_encoder_output_dim(self):
        with torch.no_grad():                                          # 禁止梯度计算
            dummy = torch.zeros(1, self.channels, self.resolution, self.resolution)  # 构造一个假输入
            out = self.encoder(dummy)                                  # 通过编码器前向传播
        return out.shape[1]                                            # 返回展平后的维度

    # 构建鉴别器，将图像映射到类别得分（不加 softmax，因为交叉熵损失内部会处理）
    def build_discriminator(self, min_latent_res=8):
        self.discriminator_encoder = self.build_common_encoder(min_latent_res)  # 使用通用编码器构建鉴别器部分
        encoder_output_dim = self._get_encoder_output_dim()            # 获取编码器输出维度
        self.discriminator_fc = nn.Linear(encoder_output_dim, self.nclasses + 1)  # 全连接层输出类别数+1（最后一类代表伪造图像）

    # 接口函数：根据类别生成图像（先生成潜在向量再生成图像）
    def generate(self, c, bg=None):
        latent = self.generate_latent(c, bg)                           # 根据类别生成潜在向量
        latent = torch.tensor(latent, dtype=torch.float32).to(self.device)  # 转为张量并移动到设备上
        with torch.no_grad():                                          # 禁止梯度计算
            generated_img = self.generate_from_latent(latent)          # 生成图像
        return generated_img                                           # 返回生成的图像

    # 根据类别采样潜在向量（使用多变量正态分布，每个类别有独立均值与协方差）
    def generate_latent(self, c, bg=None, n_mix=10):
        latent_list = []                                               # 存储采样的潜在向量
        for e in c:                                                    # 对于每个类别标签
            z = np.random.multivariate_normal(self.means[e], self.covariances[e])  # 从多变量正态分布采样
            latent_list.append(z)                                      # 添加采样结果
        return np.array(latent_list)                                   # 返回潜在向量数组

    # 鉴别器前向传播：将图像输入编码器后，再通过全连接层获得类别得分
    def discriminate(self, image):
        features = self.discriminator_encoder(image)                   # 提取图像特征
        logits = self.discriminator_fc(features)                       # 计算类别得分
        return logits                                                  # 返回得分（后续交叉熵损失会自动计算 softmax）

    # 类初始化函数
    def __init__(self, classes, target_class_id,
                 dratio_mode="uniform", gratio_mode="uniform",
                 adam_lr=0.00005, latent_size=100,
                 res_dir="./res-tmp", image_shape=[3, 32, 32], min_latent_res=8, device='cpu'):
        self.device = device                                         # 设置运算设备（如 'cpu' 或 'cuda'）
        self.gratio_mode = gratio_mode                               # 生成器采样模式
        self.dratio_mode = dratio_mode                               # 鉴别器采样模式
        self.classes = classes                                       # 类别列表
        self.target_class_id = target_class_id                       # 目标类别 ID
        self.nclasses = len(classes)                                 # 类别数量
        self.latent_size = latent_size                               # 潜在向量维度
        self.res_dir = res_dir                                       # 结果保存目录
        self.image_shape = image_shape                               # 图像形状
        self.channels = image_shape[0]                               # 图像通道数
        self.resolution = image_shape[1]                             # 图像分辨率（假设图像为正方形）
        if image_shape[1] != image_shape[2]:                         # 检查图像是否为正方形
            print("Error: only squared images currently supported by balancingGAN")  # 输出错误信息
            exit(1)                                                 # 退出程序
        self.min_latent_res = min_latent_res                         # 最小潜在分辨率
        self.adam_lr = adam_lr                                       # Adam 优化器学习率
        self.adam_beta_1 = 0.5                                       # Adam 优化器 beta1 参数
        self.train_history = defaultdict(list)                      # 初始化训练历史记录
        self.test_history = defaultdict(list)                       # 初始化测试历史记录
        self.trained = False                                         # 标记是否完成训练

        # 构建生成器网络
        self.build_generator(latent_size, init_resolution=min_latent_res)  # 构建生成器
        self.gen_fc.to(self.device)                                  # 将全连接部分移动到指定设备
        self.gen_conv.to(self.device)                                # 将卷积部分移动到设备
        # 先构建重构器（从而定义 self.encoder）
        self.build_reconstructor(latent_size, min_latent_res=min_latent_res)
        # 再构建鉴别器
        self.build_discriminator(min_latent_res=min_latent_res)

        # 构建鉴别器网络
        #self.build_discriminator(min_latent_res=min_latent_res)      # 构建鉴别器
        self.discriminator_encoder.to(self.device)                   # 移动鉴别器编码器到设备
        self.discriminator_fc.to(self.device)                        # 移动鉴别器全连接层到设备

        # 构建重构器网络
        #self.build_reconstructor(latent_size, min_latent_res=min_latent_res)  # 构建重构器
        self.encoder.to(self.device)                                 # 移动编码器到设备
        self.reconstructor_fc.to(self.device)                        # 移动全连接重构器到设备

        # 定义各个部分的优化器
        self.optimizer_G = optim.Adam(list(self.gen_fc.parameters()) + list(self.gen_conv.parameters()),
                                      lr=self.adam_lr, betas=(self.adam_beta_1, 0.999))  # 生成器优化器
        self.optimizer_D = optim.Adam(list(self.discriminator_encoder.parameters()) + list(self.discriminator_fc.parameters()),
                                      lr=self.adam_lr, betas=(self.adam_beta_1, 0.999))  # 鉴别器优化器
        self.optimizer_R = optim.Adam(list(self.encoder.parameters()) + list(self.reconstructor_fc.parameters()),
                                      lr=self.adam_lr, betas=(self.adam_beta_1, 0.999))  # 重构器优化器

    # 根据目标分布采样标签，返回采样的标签数组
    def _biased_sample_labels(self, samples, target_distribution="uniform"):
        if target_distribution == "d":                                 # 若目标为鉴别器
            distribution = self.class_dratio                          # 使用鉴别器分布
        elif target_distribution == "g":                               # 若目标为生成器
            distribution = self.class_gratio                          # 使用生成器分布
        else:
            distribution = self.class_uratio                           # 默认使用均匀分布
        sampled_labels = np.full(samples, 0)                           # 初始化标签数组
        sampled_labels_p = np.random.uniform(0, 1, samples)            # 生成 [0,1] 均匀分布的随机数
        for c in range(self.nclasses):                                 # 对于每个类别
            mask = (sampled_labels_p > 0) & (sampled_labels_p <= distribution[c])  # 选择满足条件的样本
            sampled_labels[mask] = self.classes[c]                     # 将对应类别赋值给这些样本
            sampled_labels_p = sampled_labels_p - distribution[c]      # 更新剩余概率
        return sampled_labels                                          # 返回采样标签

    # 训练单个 epoch，包含鉴别器和生成器的训练
    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []                                           # 存储每个 batch 的鉴别器损失
        epoch_gen_loss = []                                            # 存储每个 batch 的生成器损失
        for image_batch, label_batch in bg_train.next_batch():         # 遍历训练批次（假定 bg_train 提供 next_batch() 接口）
            image_batch = torch.tensor(image_batch, dtype=torch.float32).to(self.device)  # 将图像批次转换为张量并转移至设备
            label_batch = torch.tensor(label_batch, dtype=torch.long).to(self.device)       # 将标签转换为张量
            crt_batch_size = label_batch.shape[0]                       # 当前批次样本数量

            # 训练鉴别器部分
            fake_size = int(np.ceil(crt_batch_size * 1.0 / self.nclasses))  # 计算伪造样本数
            sampled_labels = self._biased_sample_labels(fake_size, "d") # 根据分布采样伪造样本的标签
            latent_gen = self.generate_latent(sampled_labels, bg_train) # 根据采样标签生成潜在向量
            latent_gen = torch.tensor(latent_gen, dtype=torch.float32).to(self.device)  # 转换为张量
            with torch.no_grad():                                       # 禁止梯度计算生成伪造图像
                generated_images = self.generate_from_latent(latent_gen)  # 生成伪造图像
            X = torch.cat((image_batch, generated_images), dim=0)       # 将真实图像与伪造图像拼接在一起
            aux_y_real = label_batch                                  # 真实图像对应的标签
            aux_y_fake = torch.full((fake_size,), self.nclasses, dtype=torch.long).to(self.device)  # 伪造图像统一标记为最后一类
            aux_y = torch.cat((aux_y_real, aux_y_fake), dim=0)          # 拼接真实和伪造标签
            logits = self.discriminate(X)                             # 鉴别器前向传播得到得分
            loss_D = F.cross_entropy(logits, aux_y)                   # 计算交叉熵损失
            epoch_disc_loss.append(loss_D.item())                     # 记录当前 batch 鉴别器损失
            self.optimizer_D.zero_grad()                              # 清空鉴别器梯度
            loss_D.backward()                                         # 反向传播计算梯度
            self.optimizer_D.step()                                   # 更新鉴别器参数

            # 训练生成器部分
            total_fake = fake_size + crt_batch_size                   # 生成器训练所需的总样本数
            sampled_labels = self._biased_sample_labels(total_fake, "g")  # 根据生成器分布采样标签
            latent_gen = self.generate_latent(sampled_labels, bg_train) # 根据采样标签生成潜在向量
            latent_gen = torch.tensor(latent_gen, dtype=torch.float32).to(self.device)  # 转换为张量
            fake_images = self.generate_from_latent(latent_gen)         # 生成伪造图像
            logits_fake = self.discriminate(fake_images)              # 鉴别器对伪造图像进行预测
            target_labels = torch.tensor(sampled_labels, dtype=torch.long).to(self.device)  # 目标标签
            loss_G = F.cross_entropy(logits_fake, target_labels)      # 计算生成器损失
            epoch_gen_loss.append(loss_G.item())                      # 记录生成器损失
            self.optimizer_G.zero_grad()                              # 清空生成器梯度
            loss_G.backward()                                         # 反向传播计算生成器梯度
            self.optimizer_G.step()                                   # 更新生成器参数

        # 返回当前 epoch 的平均损失
        avg_disc_loss = np.mean(epoch_disc_loss)                      # 计算平均鉴别器损失
        avg_gen_loss = np.mean(epoch_gen_loss)                        # 计算平均生成器损失
        return avg_disc_loss, avg_gen_loss                           # 返回平均损失

    # 根据实际类别比例设置采样比例（均衡或重平衡模式）
    def _set_class_ratios(self):
        target = 1.0 / self.nclasses                                  # 均匀目标比例
        self.class_uratio = np.full(self.nclasses, target)             # 均匀分布比例
        self.class_gratio = np.full(self.nclasses, 0.0)                # 初始化生成器采样比例
        for c in range(self.nclasses):                                # 对每个类别
            if self.gratio_mode == "uniform":                         # 均匀模式
                self.class_gratio[c] = target
            elif self.gratio_mode == "rebalance":                     # 重平衡模式
                self.class_gratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown gmode " + self.gratio_mode)
                exit(1)
        self.class_dratio = np.full(self.nclasses, 0.0)                # 初始化鉴别器采样比例
        for c in range(self.nclasses):                                # 对每个类别
            if self.dratio_mode == "uniform":
                self.class_dratio[c] = target
            elif self.dratio_mode == "rebalance":
                self.class_dratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown dmode " + self.dratio_mode)
                exit(1)
        # 如果重平衡后比例为负，则归零并重新归一化
        if self.gratio_mode == "rebalance":
            self.class_gratio[self.class_gratio < 0] = 0
            self.class_gratio = self.class_gratio / np.sum(self.class_gratio)
        if self.dratio_mode == "rebalance":
            self.class_dratio[self.class_dratio < 0] = 0
            self.class_dratio = self.class_dratio / np.sum(self.class_dratio)

    # 初始化自动编码器（预训练生成器和重构器）
    def init_autoenc(self, bg_train, gen_fname=None, rec_fname=None):
        if gen_fname is None:
            generator_fname = os.path.join(self.res_dir, "{}_decoder.pth".format(self.target_class_id))
        else:
            generator_fname = gen_fname
        if rec_fname is None:
            reconstructor_fname = os.path.join(self.res_dir, "{}_encoder.pth".format(self.target_class_id))
        else:
            reconstructor_fname = rec_fname
        multivariate_prelearnt = False                             # 标记是否预加载多变量分布参数
        if os.path.exists(generator_fname) and os.path.exists(reconstructor_fname):
            print("BAGAN: loading autoencoder: ", generator_fname, reconstructor_fname)
            gen_state = torch.load(generator_fname, map_location=self.device)  # 加载生成器权重
            self.gen_fc.load_state_dict(gen_state['gen_fc'])         # 加载全连接层参数
            self.gen_conv.load_state_dict(gen_state['gen_conv'])     # 加载卷积层参数
            rec_state = torch.load(reconstructor_fname, map_location=self.device)  # 加载重构器权重
            self.encoder.load_state_dict(rec_state['encoder'])       # 加载编码器参数
            self.reconstructor_fc.load_state_dict(rec_state['reconstructor_fc'])  # 加载全连接重构器参数
            means_path = os.path.join(self.res_dir, "{}_means.npy".format(self.target_class_id))
            cov_path = os.path.join(self.res_dir, "{}_covariances.npy".format(self.target_class_id))
            if os.path.exists(means_path) and os.path.exists(cov_path):
                multivariate_prelearnt = True
                print("BAGAN: loading multivariate: ", cov_path, means_path)
                self.covariances = np.load(cov_path)                 # 加载协方差矩阵
                self.means = np.load(means_path)                     # 加载均值向量
        else:
            print("BAGAN: training autoencoder")
            autoenc_train_loss = []                                  # 记录自动编码器训练损失
            for e in range(self.autoenc_epochs):
                print('Autoencoder train epoch: {}/{}'.format(e+1, self.autoenc_epochs))
                autoenc_train_loss_crt = []                          # 当前 epoch 损失列表
                for image_batch, label_batch in bg_train.next_batch():
                    image_batch = torch.tensor(image_batch, dtype=torch.float32).to(self.device)
                    # reconstructed = self.generate_from_latent(self.encoder(image_batch))  # 重构图像（此处简单示例）
                    latent = self.reconstructor_fc(self.encoder(image_batch))
                    reconstructed = self.generate_from_latent(latent)
                    loss = F.mse_loss(reconstructed, image_batch)    # 均方误差损失
                    self.optimizer_R.zero_grad()                     # 清空重构器梯度
                    loss.backward()                                  # 反向传播
                    self.optimizer_R.step()                          # 更新重构器参数
                    autoenc_train_loss_crt.append(loss.item())       # 记录损失
                autoenc_train_loss.append(np.mean(autoenc_train_loss_crt))
            autoenc_loss_fname = os.path.join(self.res_dir, "{}_autoencoder.csv".format(self.target_class_id))
            with open(autoenc_loss_fname, 'w') as csvfile:
                for item in autoenc_train_loss:
                    csvfile.write("%s\n" % item)                     # 将损失写入 CSV 文件
            torch.save({'gen_fc': self.gen_fc.state_dict(), 'gen_conv': self.gen_conv.state_dict()}, generator_fname)  # 保存生成器权重
            torch.save({'encoder': self.encoder.state_dict(), 'reconstructor_fc': self.reconstructor_fc.state_dict()}, reconstructor_fname)  # 保存重构器权重
        # 如果没有预加载多变量分布，则计算各类别的均值与协方差
        if not multivariate_prelearnt:
            print("BAGAN: computing multivariate")
            self.covariances = []
            self.means = []
            for c in range(self.nclasses):
                imgs = bg_train.dataset_x[bg_train.per_class_ids[c]]  # 获取该类别样本
                imgs = torch.tensor(imgs, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    latent = self.encoder(imgs)                    # 得到编码器输出
                latent_np = latent.cpu().numpy()
                self.covariances.append(np.cov(latent_np, rowvar=False))  # 计算协方差
                self.means.append(np.mean(latent_np, axis=0))      # 计算均值
            self.covariances = np.array(self.covariances)
            self.means = np.array(self.means)
            cov_path = os.path.join(self.res_dir, "{}_covariances.npy".format(self.target_class_id))
            means_path = os.path.join(self.res_dir, "{}_means.npy".format(self.target_class_id))
            print("BAGAN: saving multivariate: ", cov_path, means_path)
            np.save(cov_path, self.covariances)                      # 保存协方差矩阵
            np.save(means_path, self.means)                          # 保存均值向量
            print("BAGAN: saved multivariate")

    # 辅助函数：获取最新备份的文件名
    def _get_lst_bck_name(self, element):
        files = [f for f in os.listdir(self.res_dir) if re.match(r'bck_c_{}'.format(self.target_class_id) + "_" + element, f)]
        if len(files) > 0:
            fname = files[0]
            e_str = os.path.splitext(fname)[0].split("_")[-1]
            epoch = int(e_str)
            return epoch, fname
        else:
            return 0, None

    # 初始化 GAN 模型（加载备份权重）
    def init_gan(self):
        epoch, generator_fname = self._get_lst_bck_name("generator")
        new_e, discriminator_fname = self._get_lst_bck_name("discriminator")
        if new_e != epoch:
            return 0
        try:
            gen_state = torch.load(os.path.join(self.res_dir, generator_fname), map_location=self.device)
            self.gen_fc.load_state_dict(gen_state['gen_fc'])
            self.gen_conv.load_state_dict(gen_state['gen_conv'])
            disc_state = torch.load(os.path.join(self.res_dir, discriminator_fname), map_location=self.device)
            self.discriminator_encoder.load_state_dict(disc_state['discriminator_encoder'])
            self.discriminator_fc.load_state_dict(disc_state['discriminator_fc'])
            return epoch
        except:
            return 0

    # 备份当前模型参数
    def backup_point(self, epoch):
        _, old_bck_g = self._get_lst_bck_name("generator")
        _, old_bck_d = self._get_lst_bck_name("discriminator")
        try:
            os.remove(os.path.join(self.res_dir, old_bck_g))
            os.remove(os.path.join(self.res_dir, old_bck_d))
        except:
            pass
        generator_fname = os.path.join(self.res_dir, "bck_c_{}_generator_e_{}.pth".format(self.target_class_id, epoch))
        discriminator_fname = os.path.join(self.res_dir, "bck_c_{}_discriminator_e_{}.pth".format(self.target_class_id, epoch))
        torch.save({'gen_fc': self.gen_fc.state_dict(), 'gen_conv': self.gen_conv.state_dict()}, generator_fname)
        torch.save({'discriminator_encoder': self.discriminator_encoder.state_dict(), 'discriminator_fc': self.discriminator_fc.state_dict()}, discriminator_fname)

    # 主训练函数，训练 GAN 模型
    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            self.autoenc_epochs = epochs                          # 自动编码器训练轮数
            self.class_aratio = bg_train.get_class_probability()   # 获取各类别实际比例
            self._set_class_ratios()                               # 设置类别平衡比例
            print("uratio set to: {}".format(self.class_uratio))
            print("dratio set to: {}".format(self.class_dratio))
            print("gratio set to: {}".format(self.class_gratio))
            print("BAGAN init_autoenc")
            self.init_autoenc(bg_train)                            # 初始化自动编码器
            print("BAGAN autoenc initialized, init gan")
            start_e = self.init_gan()                              # 初始化 GAN（加载备份）
            print("BAGAN gan initialized, start_e: ", start_e)
            # 初始时生成部分样本并保存图像（此处仅为示例，实际可根据需求扩展）
            crt_c = 0
            act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            act_img_samples = torch.tensor(act_img_samples, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                rec_samples = self.generate(self.classes[crt_c], bg_train)
                fake_samples = self.generate_samples(crt_c, 10, bg_train)
            save_image_array(np.array([act_img_samples.cpu().numpy()]), '{}/cmp_class_{}_init.png'.format(self.res_dir, self.target_class_id))
            # 开始正式训练 GAN
            for e in range(start_e, epochs):
                print('GAN train epoch: {}/{}'.format(e+1, epochs))
                train_disc_loss, train_gen_loss = self._train_one_epoch(bg_train)
                # 测试阶段：生成伪造图像并计算测试损失
                nb_test = bg_test.get_num_samples()
                fake_size = int(np.ceil(nb_test * 1.0 / self.nclasses))
                sampled_labels = self._biased_sample_labels(nb_test, "d")
                latent_gen = self.generate_latent(sampled_labels, bg_test)
                latent_gen = torch.tensor(latent_gen, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    generated_images = self.generate_from_latent(latent_gen)
                X = torch.cat((torch.tensor(bg_test.dataset_x, dtype=torch.float32).to(self.device), generated_images), dim=0)
                aux_y_real = torch.tensor(bg_test.dataset_y, dtype=torch.long).to(self.device)
                aux_y_fake = torch.full((fake_size,), self.nclasses, dtype=torch.long).to(self.device)
                aux_y = torch.cat((aux_y_real, aux_y_fake), dim=0)
                logits = self.discriminate(X)
                test_disc_loss = F.cross_entropy(logits, aux_y).item()
                total_fake = fake_size + nb_test
                sampled_labels = self._biased_sample_labels(total_fake, "g")
                latent_gen = self.generate_latent(sampled_labels, bg_test)
                latent_gen = torch.tensor(latent_gen, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    logits_fake = self.discriminate(self.generate_from_latent(latent_gen))
                target_labels = torch.tensor(sampled_labels, dtype=torch.long).to(self.device)
                test_gen_loss = F.cross_entropy(logits_fake, target_labels).item()
                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                self.test_history['disc_loss'].append(test_disc_loss)
                self.test_history['gen_loss'].append(test_gen_loss)
                print("train_disc_loss {},\ttrain_gen_loss {},\ttest_disc_loss {},\ttest_gen_loss {}".format(
                    train_disc_loss, train_gen_loss, test_disc_loss, test_gen_loss
                ))
                # 每 10 个 epoch 保存一次生成样本图像
                if e % 10 == 9:
                    img_samples = []
                    for c in range(0, self.nclasses):
                        samples = self.generate_samples(c, 10, bg_train)
                        img_samples.append(samples.cpu().numpy())
                    img_samples = np.array(img_samples)
                    save_image_array(img_samples, '{}/plot_class_{}_epoch_{}.png'.format(self.res_dir, self.target_class_id, e))
                # 每 10 个 epoch 备份一次模型并保存比较图像
                if e % 10 == 5:
                    self.backup_point(e)
                    img_samples = []
                    for crt_c in range(0, self.nclasses):
                        act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        act_img_samples = torch.tensor(act_img_samples, dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            rec_samples = self.generate(self.classes[crt_c], bg_train)
                            fake_samples = self.generate_samples(crt_c, 10, bg_train)
                        combined = torch.stack([act_img_samples, rec_samples, fake_samples], dim=0)
                        img_samples.append(combined.cpu().numpy())
                    img_samples = np.array(img_samples)
                    save_image_array(img_samples, '{}/cmp_class_{}_epoch_{}.png'.format(self.res_dir, self.target_class_id, e))
            self.trained = True

    # 根据类别生成样本图像的接口
    def generate_samples(self, c, samples, bg=None):
        labels = np.full(samples, self.classes[c])                    # 构造指定类别的标签数组
        return self.generate(labels, bg)                                # 调用 generate 函数生成图像

    # 保存训练历史和模型参数
    def save_history(self, res_dir, class_id):
        if self.trained:
            filename = os.path.join(res_dir, "class_{}_score.csv".format(class_id))
            generator_fname = os.path.join(res_dir, "class_{}_generator.pth".format(class_id))
            discriminator_fname = os.path.join(res_dir, "class_{}_discriminator.pth".format(class_id))
            reconstructor_fname = os.path.join(res_dir, "class_{}_reconstructor.pth".format(class_id))
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['train_gen_loss', 'train_disc_loss', 'test_gen_loss', 'test_disc_loss']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for e in range(len(self.train_history['gen_loss'])):
                    row = {
                        'train_gen_loss': self.train_history['gen_loss'][e],
                        'train_disc_loss': self.train_history['disc_loss'][e],
                        'test_gen_loss': self.test_history['gen_loss'][e],
                        'test_disc_loss': self.test_history['disc_loss'][e]
                    }
                    writer.writerow(row)
            torch.save({'gen_fc': self.gen_fc.state_dict(), 'gen_conv': self.gen_conv.state_dict()}, generator_fname)
            torch.save({'discriminator_encoder': self.discriminator_encoder.state_dict(), 'discriminator_fc': self.discriminator_fc.state_dict()}, discriminator_fname)
            torch.save({'encoder': self.encoder.state_dict(), 'reconstructor_fc': self.reconstructor_fc.state_dict()}, reconstructor_fname)

    # 加载预训练模型参数
    def load_models(self, fname_generator, fname_discriminator, fname_reconstructor, bg_train=None):
        self.init_autoenc(bg_train, gen_fname=fname_generator, rec_fname=fname_reconstructor)
        disc_state = torch.load(fname_discriminator, map_location=self.device)
        self.discriminator_encoder.load_state_dict(disc_state['discriminator_encoder'])
        self.discriminator_fc.load_state_dict(disc_state['discriminator_fc'])
