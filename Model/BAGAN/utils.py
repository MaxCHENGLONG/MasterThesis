import numpy as np  # 导入NumPy库，用于数组操作
from PIL import Image  # 从PIL库导入Image模块，用于图像处理


def save_image_array(img_array, fname):
    # 定义函数 save_image_array，用于将图像数组保存到指定的文件名中
    channels = img_array.shape[2]  # 获取图像数组中小图的通道数（例如：1代表灰度图，3代表彩色图）
    resolution = img_array.shape[-1]  # 获取每个小图的分辨率（这里假定宽度和高度相同，取最后一个维度）
    img_rows = img_array.shape[0]  # 获取网格中小图的行数
    img_cols = img_array.shape[1]  # 获取网格中小图的列数

    # 创建一个空的图像数组，用于拼接所有小图，数组大小为[channels, resolution * img_rows, resolution * img_cols]
    img = np.full([channels, resolution * img_rows, resolution * img_cols], 0.0)
    
    # 遍历每一行的小图
    for r in range(img_rows):
        # 遍历每一列的小图
        for c in range(img_cols):
            # 将当前位置的小图复制到大图中对应的位置
            # 注意：这里对列索引使用了(c % 10)，即将列数固定为10个周期排列
            img[:,
                (resolution * r): (resolution * (r + 1)),            # 指定大图中当前行对应的像素范围
                (resolution * (c % 10)): (resolution * ((c % 10) + 1))  # 指定大图中当前列对应的像素范围
                ] = img_array[r, c]

    # 将图像像素值从[-1, 1]映射到[0, 255]，并转换为无符号8位整数（uint8），便于显示和保存
    img = (img * 127.5 + 127.5).astype(np.uint8)
    
    # 如果图像只有1个通道，则去掉通道维度，使图像变为二维
    if (img.shape[0] == 1):
        img = img[0]
    else:
        # 否则，将通道维度从第一个位置移动到最后，转换为HWC格式（高度×宽度×通道）
        img = np.rollaxis(img, 0, 3)

    # 使用PIL将NumPy数组转换为图像对象，并保存到指定文件名
    Image.fromarray(img).save(fname)
