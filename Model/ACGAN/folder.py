import torch.utils.data as data  # 导入 PyTorch 中的数据工具模块，并命名为 data

from PIL import Image  # 从 PIL 库中导入 Image 模块，用于图像操作
import os  # 导入 os 模块，用于文件与目录的操作
import os.path  # 导入 os.path 模块，用于处理文件路径

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']  # 定义支持的图像文件扩展名列表

def is_image_file(filename):  # 定义函数 is_image_file，判断给定文件名是否为图像文件
    """Checks if a file is an image.
    检查文件是否为图像.

    Args:
        filename (string): path to a file
            filename (字符串): 文件的路径

    Returns:
        bool: True if the filename ends with a known image extension
              如果文件名以已知图像扩展名结束，则返回 True
    """
    filename_lower = filename.lower()  # 将文件名转换成小写，便于比较扩展名
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)  # 遍历扩展名列表，若匹配则返回 True，否则返回 False

def find_classes(dir, classes_idx=None):  # 定义函数 find_classes，用于查找目录中所有类别
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]  # 遍历目录中所有文件，筛选出子目录作为类别
    classes.sort()  # 将类别按字母顺序排序
    if classes_idx is not None:  # 如果指定了类别索引范围
        assert type(classes_idx) == tuple  # 断言 classes_idx 类型为元组
        start, end = classes_idx  # 解包元组，获得起始和结束索引
        classes = classes[start:end]  # 根据索引范围裁剪类别列表
    class_to_idx = {classes[i]: i for i in range(len(classes))}  # 生成类别名称到类别索引的映射字典
    return classes, class_to_idx  # 返回类别列表和映射字典

def make_dataset(dir, class_to_idx):  # 定义函数 make_dataset，用于构建图像数据集列表
    images = []  # 初始化图像数据列表
    dir = os.path.expanduser(dir)  # 扩展用户目录符号 (~) 为绝对路径
    for target in sorted(os.listdir(dir)):  # 遍历目录中排序后的子文件或子目录
        if target not in class_to_idx:  # 如果当前目标不在分类字典中则跳过
            continue  # 继续下一轮循环
        d = os.path.join(dir, target)  # 构造目标目录的完整路径
        if not os.path.isdir(d):  # 如果 d 不是一个目录，则跳过
            continue  # 继续下一轮循环

        for root, _, fnames in sorted(os.walk(d)):  # 遍历目录 d 下的所有子目录及文件（使用 os.walk）
            for fname in sorted(fnames):  # 遍历排序后的文件名列表
                if is_image_file(fname):  # 如果文件名符合图像扩展名标准
                    path = os.path.join(root, fname)  # 构造文件的完整路径
                    item = (path, class_to_idx[target])  # 构造包括图像路径和对应类别索引的元组
                    images.append(item)  # 将该元组添加到图像数据列表中

    return images  # 返回包含所有图像和类别索引的列表

def pil_loader(path):  # 定义函数 pil_loader，使用 PIL 加载图像
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # 以文件方式打开路径以避免资源警告（参见 Pillow issue 835）
    with open(path, 'rb') as f:  # 以二进制模式打开图像文件
        with Image.open(f) as img:  # 使用 PIL 打开文件为图像对象
            return img.convert('RGB')  # 将图像转换为 RGB 模式并返回

def accimage_loader(path):  # 定义函数 accimage_loader，使用 accimage 库加载图像
    import accimage  # 导入 accimage 库
    try:
        return accimage.Image(path)  # 尝试使用 accimage 加载图像
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        # 如果发生 IO 错误（可能是解码问题），则回退到使用 PIL 加载图像
        return pil_loader(path)  # 返回使用 pil_loader 加载的图像

def default_loader(path):  # 定义函数 default_loader，根据设置选择图像加载器
    from torchvision import get_image_backend  # 从 torchvision 导入函数 get_image_backend，用于检查当前图像后端
    if get_image_backend() == 'accimage':  # 如果当前图像后端为 accimage
        return accimage_loader(path)  # 则使用 accimage_loader 加载图像
    else:
        return pil_loader(path)  # 否则使用 pil_loader 加载图像

class ImageFolder(data.Dataset):  # 定义 ImageFolder 类，继承自 data.Dataset，用于加载文件夹内图像数据集
    """A generic data loader where the images are arranged in this way: ::
    一个通用的数据加载器，图像组织结构如下： ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
            根目录路径
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
            图像变换函数（可选），接收一个 PIL 图像并返回变换后结果，例如 transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            目标变换函数（可选），用于转换标签
        loader (callable, optional): A function to load an image given its path.
            图像加载函数（可选），通过图像路径加载图像

     Attributes:
        classes (list): List of the class names.
            类别名称列表
        class_to_idx (dict): Dict with items (class_name, class_index).
            类别名称到类别索引的映射字典
        imgs (list): List of (image path, class_index) tuples
            图像路径和类别索引元组列表
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, classes_idx=None):  # 构造函数：初始化 ImageFolder 实例
        self.classes_idx = classes_idx  # 保存传入的类别索引范围（如果有）
        classes, class_to_idx = find_classes(root, self.classes_idx)  # 查找类别并获得类别映射
        imgs = make_dataset(root, class_to_idx)  # 构造数据集，获取图像路径和对应的类别索引
        if len(imgs) == 0:  # 如果没有找到任何图像
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
            # 抛出运行时错误，并提示支持的图像文件扩展名

        self.root = root  # 保存根目录路径
        self.imgs = imgs  # 保存图像数据列表
        self.classes = classes  # 保存类别列表
        self.class_to_idx = class_to_idx  # 保存类别到索引的映射字典
        self.transform = transform  # 保存图像变换函数
        self.target_transform = target_transform  # 保存标签变换函数
        self.loader = loader  # 保存图像加载函数

    def __getitem__(self, index):  # 定义 __getitem__ 方法，使 ImageFolder 支持下标操作
        """
        Args:
            index (int): Index
                索引值（整数）
        Returns:
            tuple: (image, target) where target is class_index of the target class.
                   返回一个元组：(图像, 标签)，其中标签为目标类别的索引
        """
        path, target = self.imgs[index]  # 根据索引获取图像路径和对应的目标类别
        img = self.loader(path)  # 使用指定的加载器加载图像
        if self.transform is not None:  # 如果指定了图像变换函数，则进行图像转换
            img = self.transform(img)  # 应用图像变换
        if self.target_transform is not None:  # 如果指定了标签变换函数，则进行标签转换
            target = self.target_transform(target)  # 应用标签变换

        return img, target  # 返回处理后的图像及其目标类别

    def __len__(self):  # 定义 __len__ 方法，返回数据集大小
        return len(self.imgs)  # 数据集大小即图像列表的长度