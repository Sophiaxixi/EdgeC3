import os
import sys
import numpy as np
import math


from util.utils import get_index_from_one_hot_label, get_even_odd_from_one_hot_label

def get_stream_data(dataset, total_data, dataset_file_path, data_size, sim_round=None):

    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        from data_reader.mnist_extractor import mnist_extract

        if total_data > 60000:
            total_data_train = 60000 # 训练的最多60000
        else:
            total_data_train = total_data # 48000

        if total_data > 10000:
            total_data_test = 10000 # test的最多10000 # 10000
        else:
            total_data_test = total_data

        if sim_round is None:
            start_index_train = 0
            start_index_test = 0
        else:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))

        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path)
        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path)

        # print("label length is:" + str(len(train_label))) # label length is:48000
        # print(train_label[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
        # # 完整的label,每一个label是 array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        # print("image length is:" + str(len(train_image))) # image length is:48000
        # print(train_image[0])
        # # [ 0.         0.         0.01176471 0.07058824 0.07058824 0.07058824........]
        # print(train_image[0].shape) # (784,)

        # 下面把train_label的值改了，原来label为偶数的改成1，奇数改成-1，改之前要把原始的留下来
        train_label_orig = []
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label)

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

            for i in range(0, len(test_label)):
                test_label[i] = get_even_odd_from_one_hot_label(test_label[i])

        indicates_train = [i for i in range(0, len(train_label))]  # 48000个
        np.random.RandomState().shuffle(indicates_train)# 把seed=1000去掉 否则每次shuffle之后的数都一样 tau没影响
        indicates_train = indicates_train[0:data_size]  # 数目是D[0]
        print("indicates_train is")
        print(indicates_train)
        stream_train_image = [train_image[i] for i in indicates_train]
        stream_train_label = [train_label[i] for i in indicates_train]
        stream_train_label_orig = [train_label_orig[i] for i in indicates_train]

        # test的数目不变

        # indicates_test = [i for i in range(0, len(test_label))]  # 10000个
        # np.random.RandomState(seed=1000).shuffle(indicates_test)
        # test_data_size = math.floor(data_size/4)
        # indicates_test= indicates_test[0:test_data_size]  # test数目始终是train的1/4: D[0]/4
        # stream_test_image = [test_image[i] for i in indicates_test]
        # stream_test_label = [test_label[i] for i in indicates_test]

    elif dataset == 'CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label)

        indicates_train = [i for i in range(0, len(train_label))]  # 48000个
        np.random.RandomState().shuffle(indicates_train)
        indicates_train = indicates_train[0:data_size]  # 数目是D[0]
        print("indicates_train is")
        print(indicates_train)
        stream_train_image = [train_image[i] for i in indicates_train]
        stream_train_label = [train_label[i] for i in indicates_train]
        stream_train_label_orig = [train_label_orig[i] for i in indicates_train]
    else:
        raise Exception('Unknown dataset name.')

    return stream_train_image, stream_train_label, test_image, test_label, stream_train_label_orig

def get_data(dataset, total_data, dataset_file_path=os.path.dirname(__file__), sim_round=None):
    from data_reader.sarrt_extractor import sarrt_extract

    if dataset == 'SAR_RT':
        # total_data = 26640
        total_data_train = 21396
        total_data_test = 5244

        train_image, train_label = sarrt_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = sarrt_extract(0, total_data_test, False, dataset_file_path)

        # train_label_orig must be determined before the values in train_label are overwritten below
        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label)

        return train_image, train_label, test_image, test_label, train_label_orig


    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        from data_reader.mnist_extractor import mnist_extract

        if total_data > 60000:
            total_data_train = 60000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        if sim_round is None:
            start_index_train = 0
            start_index_test = 0
        else:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))

        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path)
        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path)

        # train_label_orig must be determined before the values in train_label are overwritten below
        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label)

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(test_label)):
                test_label[i] = get_even_odd_from_one_hot_label(test_label[i])

    elif dataset == 'CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label)

    else:
        raise Exception('Unknown dataset name.')

    return train_image, train_label, test_image, test_label, train_label_orig