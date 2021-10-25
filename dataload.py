"""
Author: Zhao Jun
Date: 2021.3.17
Dataload for CWRU

"""

from scipy.io import loadmat
import numpy as np
import os
from scipy.fftpack import fft
import matplotlib.pylab as plt


def capture(original_path):
    """读取mat文件，返回字典

    :param original_path: 读取路径
    :return: 数据字典
    """
    files = {}
    for i in filenames:
        # 文件路径
        file_path = os.path.join(original_path, i)
        file = loadmat(file_path)
        file_keys = file.keys()
        for key in file_keys:
            if 'DE' in key:
                files[i] = file[key].ravel()
    return files


def slice_enc(data, number_train, number_val, length):
    """将数据切分为前面多少比例，后面多少比例.

    :param data: 单挑数据
    :param slice_rate: 验证集以及测试集所占的比例
    :return: 切分好的数据
    """
    keys = data.keys()
    dataset_train = np.empty([len(keys), number_train, int(length / 2)])
    dataset_val = np.empty([len(keys), number_val, int(length / 2)])
    label_train = np.empty([len(keys), num_train])
    label_val = np.empty([len(keys), num_val])
    k = 0
    # 按照title划分数据
    for i in keys:
        slice_data = data[i]
        all_lenght = len(slice_data)
        sub_dataset_train = np.empty([number_train, int(length / 2)])
        sub_dataset_val = np.empty([number_val, int(length / 2)])

        for m in range(number_train):
            num = np.random.randint(low=0, high=int(all_lenght*0.7-length))

            sub_dataset_train[m] = abs(fft(slice_data[num:num+length]))[:int(length / 2)]


        for j in range(number_val):
            num_1 = np.random.randint(low=int(all_lenght*0.7), high=all_lenght-length)

            sub_dataset_val[j] = abs(fft(slice_data[num_1:num_1+length]))[:int(length / 2)]

        dataset_val[k] = sub_dataset_val
        label_train[k] = k * np.ones(num_train)
        label_val[k] = k * np.ones(num_val)
        dataset_train[k] = sub_dataset_train
        k += 1

    return dataset_train, dataset_val, label_train, label_val


if __name__ == '__main__':
#    if not os.path.exists('foldername'):
#       os.mkdir('foldername')

    num_train = 100
    num_val = 80

    for t in range(4):

        filenames = os.listdir(r'C:\Users\yxh\Desktop\人工智能\fault diagnosis  Code\CWRUdata\data\\' + str(t) + 'HP')
        file_data = capture(r'C:\Users\yxh\Desktop\人工智能\fault diagnosis  Code\CWRUdata\data\\' + str(t) + 'HP')
        dataset_train, dataset_val, label_train, label_val = slice_enc(file_data, number_train=num_train,
                                                                       number_val=num_val, length=2048)

        np.savez(r'G:\bearing numpy data\dataset_train_' + str(t) + 'HP_' + str(num_train) + '.npz', data=dataset_train,
                 label=label_train)
        np.savez(r'G:\bearing numpy data\dataset_val_' + str(t) + 'HP_' + str(num_val) + '.npz', data=dataset_val
                 , label=label_val)
