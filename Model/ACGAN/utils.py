# 对netG和netD调用的自定义权重初始化函数
def weights_init(m):
    # 获取当前模块m的类名
    classname = m.__class__.__name__
    # 如果类名中包含 'Conv'（卷积层）字样，则初始化卷积层权重
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # 用均值为0.0，标准差为0.02的正态分布初始化卷积层权重
    # 如果类名中包含 'BatchNorm'（批归一化层）字样，则初始化批归一化层参数
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # 用均值为1.0，标准差为0.02的正态分布初始化批归一化层权重
        m.bias.data.fill_(0)              # 将批归一化层的偏置初始化为0

# 计算当前分类准确率的函数
def compute_acc(preds, labels):
    correct = 0  # 初始化正确预测的计数器为0
    preds_ = preds.data.max(1)[1]  # 获取每个预测样本中概率最大的索引，即预测的类别
    correct = preds_.eq(labels.data).cpu().sum()  # 将预测类别与真实标签比较，统计相等的数量（正确预测数）
    acc = float(correct) / float(len(labels.data)) * 100.0  # 计算准确率：正确预测数除以总样本数，并转换为百分比
    return acc # 返回准确率