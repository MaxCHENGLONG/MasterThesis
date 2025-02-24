"""
(C) Copyright IBM Corporation 2018
所有权利保留。该程序及其伴随材料均根据 Eclipse Public License v1.0 提供，
许可证内容见 http://www.eclipse.org/legal/epl-v10.html
"""

from collections import defaultdict  # 导入 defaultdict，用于统计记录
import numpy as np  # 导入 numpy 库
from optparse import OptionParser  # 导入 OptionParser 用于解析命令行参数
import os  # 导入 os 模块，用于文件及目录操作

# 导入基于 PyTorch 实现的 BAGAN 模型（请确保该文件已使用 PyTorch 重写）
import Balancing_GAN as bagan  

# 导入批次数据生成器（该部分代码不涉及框架转换，保持不变）
from rw.batch_generator import BatchGenerator as BatchGenerator  

# 导入图像保存函数，用于保存生成结果图像
from utils import save_image_array  

# 主程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    argParser = OptionParser()  # 实例化 OptionParser 对象

    # 添加参数：不平衡因子（unbalance），少数类样本数量最多为其他类样本数的 u 倍
    argParser.add_option("-u", "--unbalance", default=0.2,
                         action="store", type="float", dest="unbalance",
                         help="Unbalance factor u. The minority class has at most u * otherClassSamples instances.")  # 添加 -u 参数

    # 添加参数：随机种子，用于结果可复现
    argParser.add_option("-s", "--random_seed", default=0,
                         action="store", type="int", dest="seed",
                         help="Random seed for repeatable subsampling.")  # 添加 -s 参数

    # 添加参数：鉴别器采样模式，可选 "uniform" 或 "rebalance"
    argParser.add_option("-d", "--sampling_mode_for_discriminator", default="uniform",
                         action="store", type="string", dest="dratio_mode",
                         help="Dratio sampling mode (\"uniform\",\"rebalance\").")  # 添加 -d 参数

    # 添加参数：生成器采样模式，可选 "uniform" 或 "rebalance"
    argParser.add_option("-g", "--sampling_mode_for_generator", default="uniform",
                         action="store", type="string", dest="gratio_mode",
                         help="Gratio sampling mode (\"uniform\",\"rebalance\").")  # 添加 -g 参数

    # 添加参数：训练轮数
    argParser.add_option("-e", "--epochs", default=3,
                         action="store", type="int", dest="epochs",
                         help="Training epochs.")  # 添加 -e 参数

    # 添加参数：学习率
    argParser.add_option("-l", "--learning_rate", default=0.00005,
                         action="store", type="float", dest="adam_lr",
                         help="Training learning rate.")  # 添加 -l 参数

    # 添加参数：目标类别，如果大于或等于 0，则仅训练指定类别的模型
    argParser.add_option("-c", "--target_class", default=-1,
                         action="store", type="int", dest="target_class",
                         help="If greater or equal to 0, model trained only for the specified class.")  # 添加 -c 参数

    # 添加参数：数据集名称，可选 'MNIST' 或 'CIFAR10'
    argParser.add_option("-D", "--dataset", default='MNIST',
                         action="store", type="string", dest="dataset",
                         help="Either 'MNIST', or 'CIFAR10'.")  # 添加 -D 参数

    # 解析命令行参数
    (options, args) = argParser.parse_args()  # 解析传入参数

    # 判断不平衡因子必须大于 0 且不超过 1
    assert (options.unbalance <= 1.0 and options.unbalance > 0.0), "Data unbalance factor must be > 0 and <= 1"  # 检查参数有效性

    print("Executing BAGAN.")  # 输出提示信息

    # 设置随机种子，确保结果可复现
    np.random.seed(options.seed)  # 设置 numpy 随机种子
    unbalance = options.unbalance  # 获取不平衡因子
    gratio_mode = options.gratio_mode  # 获取生成器采样模式
    dratio_mode = options.dratio_mode  # 获取鉴别器采样模式
    gan_epochs = options.epochs  # 获取训练轮数
    adam_lr = options.adam_lr  # 获取学习率
    opt_class = options.target_class  # 获取目标类别
    batch_size = 128  # 设置批次大小
    dataset_name = options.dataset  # 获取数据集名称

    # 根据数据集名称确定图像通道数：MNIST 为 1，其他（如 CIFAR10）为 3
    channels = 1 if dataset_name == 'MNIST' else 3  # 根据数据集设置通道数
    print('Using dataset: ', dataset_name)  # 输出使用的数据集名称

    # 构造结果保存目录，并将目录名包含参数信息
    res_dir = "./res_{}_dmode_{}_gmode_{}_unbalance_{}_epochs_{}_lr_{:f}_seed_{}".format(
        dataset_name, dratio_mode, gratio_mode, unbalance, options.epochs, adam_lr, options.seed)  # 构造目录名
    if not os.path.exists(res_dir):  # 如果目录不存在
        os.makedirs(res_dir)  # 创建目录

    # 读取初始数据
    print("read input data...")  # 输出提示信息
    bg_train_full = BatchGenerator(BatchGenerator.TRAIN, batch_size,
                                   class_to_prune=None, unbalance=None, dataset=dataset_name)  # 加载完整训练数据
    bg_test = BatchGenerator(BatchGenerator.TEST, batch_size,
                             class_to_prune=None, unbalance=None, dataset=dataset_name)  # 加载测试数据

    print("input data loaded...")  # 输出提示信息

    # 获取图像形状（例如 [channels, height, width]）
    shape = bg_train_full.get_image_shape()  # 获取图像尺寸

    # 计算最小潜在分辨率，初始值为图像宽度（假设图像为正方形）
    min_latent_res = shape[-1]  # 取图像的最后一个维度（宽度）
    while min_latent_res > 8:  # 当分辨率大于 8 时
        min_latent_res = min_latent_res / 2  # 不断除以 2
    min_latent_res = int(min_latent_res)  # 转换为整数

    # 获取所有类别标签列表
    classes = bg_train_full.get_label_table()  # 获取类别标签

    # 初始化训练和测试损失、生成样本的统计信息
    gan_train_losses = defaultdict(list)  # 存储训练损失
    gan_test_losses = defaultdict(list)  # 存储测试损失
    img_samples = defaultdict(list)  # 存储生成的图像样本

    # 目标类别数组：若 opt_class >= 0，则仅使用该类别，否则使用所有类别
    target_classes = np.array(range(len(classes)))  # 所有类别标签
    if opt_class >= 0:  # 如果指定了目标类别
        min_classes = np.array([opt_class])  # 仅使用该类别
    else:
        min_classes = target_classes  # 否则使用所有类别

    # 遍历每个需要处理的类别
    for c in min_classes:
        # 若不平衡因子为 1.0 且类别 c 大于 0，并且类别 0 的模型已存在，
        # 则直接对其他类别使用类别 0 的预训练模型（通过创建符号链接实现）
        if unbalance == 1.0 and c > 0 and (
            os.path.exists("{}/class_0_score.csv".format(res_dir)) and
            os.path.exists("{}/class_0_discriminator.pth".format(res_dir)) and
            os.path.exists("{}/class_0_generator.pth".format(res_dir)) and
            os.path.exists("{}/class_0_reconstructor.pth".format(res_dir))
        ):
            # 为当前类别创建指向类别 0 模型文件的符号链接
            os.symlink("{}/class_0_score.csv".format(res_dir),
                       "{}/class_{}_score.csv".format(res_dir, c))  # 建立 score 文件的符号链接
            os.symlink("{}/class_0_discriminator.pth".format(res_dir),
                       "{}/class_{}_discriminator.pth".format(res_dir, c))  # 建立 discriminator 模型的符号链接
            os.symlink("{}/class_0_generator.pth".format(res_dir),
                       "{}/class_{}_generator.pth".format(res_dir, c))  # 建立 generator 模型的符号链接
            os.symlink("{}/class_0_reconstructor.pth".format(res_dir),
                       "{}/class_{}_reconstructor.pth".format(res_dir, c))  # 建立 reconstructor 模型的符号链接
        else:
            # 若存在不平衡，则构造仅对类别 c 进行不平衡采样的训练集
            bg_train_partial = BatchGenerator(BatchGenerator.TRAIN, batch_size,
                                              class_to_prune=c, unbalance=unbalance, dataset=dataset_name)  # 加载不平衡训练数据

            # 检查针对类别 c 的模型文件是否已经存在（使用 .pth 后缀）
            if not (
                os.path.exists("{}/class_{}_score.csv".format(res_dir, c)) and
                os.path.exists("{}/class_{}_discriminator.pth".format(res_dir, c)) and
                os.path.exists("{}/class_{}_generator.pth".format(res_dir, c)) and
                os.path.exists("{}/class_{}_reconstructor.pth".format(res_dir, c))
            ):
                # 若不存在预训练模型，则需要训练新的 GAN 模型
                print("Required GAN for class {}".format(c))  # 输出当前类别需要训练提示
                print('Class counters: ', bg_train_partial.per_class_count)  # 输出该类别样本计数信息

                # 初始化基于 PyTorch 的 BAGAN 模型
                gan = bagan.BalancingGAN(
                    target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                    adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res, device='cpu'
                )  # 创建 BAGAN 模型实例

                # 训练 GAN 模型，传入不平衡训练数据和测试数据，设置训练轮数
                gan.train(bg_train_partial, bg_test, epochs=gan_epochs)  # 训练 GAN 模型

                # 保存训练历史及模型参数
                gan.save_history(res_dir, c)  # 保存当前类别的模型及训练记录
            else:
                # 否则说明该类别的 GAN 模型已存在，直接加载预训练模型
                print("Loading GAN for class {}".format(c))  # 输出加载提示

                # 初始化 BAGAN 模型实例
                gan = bagan.BalancingGAN(
                    target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                    adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res, device='cpu'
                )  # 创建 BAGAN 模型实例

                # 加载预训练模型参数，注意此处文件后缀已修改为 .pth
                gan.load_models(
                    "{}/class_{}_generator.pth".format(res_dir, c),
                    "{}/class_{}_discriminator.pth".format(res_dir, c),
                    "{}/class_{}_reconstructor.pth".format(res_dir, c),
                    bg_train=bg_train_partial  # 该参数用于初始化每个类别对应的均值和协方差矩阵
                )  # 加载预训练模型

            # 使用训练好的 GAN 模型对当前类别生成一定数量的样本图像（例如 10 张）
            img_samples['class_{}'.format(c)] = gan.generate_samples(c=c, samples=10)  # 生成样本图像

            # 将生成的样本图像保存到文件中
            save_image_array(np.array([img_samples['class_{}'.format(c)]]),
                             '{}/plot_class_{}.png'.format(res_dir, c))  # 保存生成图像到指定路径
