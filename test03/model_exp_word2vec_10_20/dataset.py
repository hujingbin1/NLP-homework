import random

import numpy as np
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


def get_data_set(data_path, window_width, window_step, negative_sample_num):
    with open(data_path, 'r', encoding='utf-8') as file:
        document = file.read()
        document = document.replace(",", "").replace("?", "").replace(".", "").replace('"', '')
        data = document.split(" ")
        print(f"数据中共有 {len(data)} 个单词")

        # 构造词典
        vocabulary = set()
        for word in data:
            vocabulary.add(word)
        vocabulary = list(vocabulary)
        print(f"词典大小为 {len(vocabulary)}")

        # index_dict
        index_dict = dict()
        for index, word in enumerate(vocabulary):
            index_dict[word] = index

        # 开始滑动窗口，构造数据
        features = []
        labels = []
        neighbor_dict = dict()

        for start_index in range(0, len(data), window_step):
            if start_index + window_width - 1 < len(data):
                mid_index = int((start_index + start_index + window_width - 1) / 2)
                for index in range(start_index, start_index + window_width):
                    if index != mid_index:
                        feature = np.zeros((len(vocabulary), len(vocabulary)))
                        feature[index_dict[data[index]]][index_dict[data[index]]] = 1
                        feature[index_dict[data[mid_index]]][index_dict[data[mid_index]]] = 1
                        features.append(feature)
                        labels.append(1)
                        if data[mid_index] in neighbor_dict.keys():
                            neighbor_dict[data[mid_index]].add(data[index])
                        else:
                            neighbor_dict[data[mid_index]] = {data[index]}
        # 负采样
        for _ in range(negative_sample_num):
            random_word = vocabulary[random.randint(0, len(vocabulary))]
            for word in vocabulary:
                if random_word not in neighbor_dict.keys() or word not in neighbor_dict[random_word]:
                    feature = np.zeros((len(vocabulary), len(vocabulary)))
                    feature[index_dict[random_word]][index_dict[random_word]] = 1
                    feature[index_dict[word]][index_dict[word]] = 1
                    features.append(feature)
                    labels.append(0)
                    break
        # 返回dataset和词典
        return MyDataSet(features, labels), vocabulary, index_dict
