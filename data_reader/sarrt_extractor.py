import os
import sys

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from struct import *
import numpy as np
from util.utils import get_one_hot_from_label_index


def sarrt_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__)):
    if is_train:
        file_path_extended = file_path + '/sarrt/data_train'
    else:
        file_path_extended = file_path + '/sarrt/data_test'

    # 获取文件夹中所有文件的文件名列表
    image_files = [f for f in os.listdir(file_path_extended) if f.endswith('.png')]

    data = []
    labels = []

    for image_file in image_files:
        # 构造完整的图片文件路径
        image_path = os.path.join(file_path_extended, image_file)

        # 使用PIL库读取图片
        img = Image.open(image_path)

        # 调整图像大小为96x96
        new_size = (32, 32)
        resized_img = img.resize(new_size)

        # 将调整大小后的图像转换为NumPy数组
        x = np.array(resized_img)

        # # 将图片转换为NumPy数组
        # x = np.array(img)

        # 展平图片数据（根据需要调整形状）
        x = x.flatten()

        # 对图片数据进行归一化（根据需要调整归一化策略）
        x = x / 255.0

        # label处理逻辑
        label = int(image_file.split("-")[0])
        label = label - 1

        y = get_one_hot_from_label_index(label)

        data.append(x)
        labels.append(y) # 是48000个label 每个label是 array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])


    return data, labels


def sarrt_extract(start_sample_index, num_samples, is_train=True, file_path=os.path.dirname(__file__)):
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return sarrt_extract_samples(sample_list, is_train, file_path)
