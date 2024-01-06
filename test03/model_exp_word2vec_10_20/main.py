import random
from math import sqrt

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import get_data_set
from model import DNN

import yaml



def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility
    固定时间种子。由于cuDNN会自动从几种算法中寻找最适合当前配置的算法，为了使选择的算法固定，所以固定时间种子
    :param seed: 时间种子
    :return: None
    """
    torch.backends.cudnn.deterministic = True  # 解决算法本身的不确定性，设置为True 保证每次结果是一致的
    torch.backends.cudnn.benchmark = False  # 解决了算法选择的不确定性，方便复现，提升训练速度
    np.random.seed(seed)  # 按顺序产生固定的数组，如果使用相同的seed，则生成的随机数相同， 注意每次生成都要调用一次
    torch.manual_seed(seed)  # 手动设置torch的随机种子，使每次运行的随机数都一致
    random.seed(seed)
    if torch.cuda.is_available():
        # 为GPU设置唯一的时间种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, config):
    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hyper_paras'])

    device = config['device']
    epoch = 0
    while epoch < config['n_epochs']:
        model.train()  # set model to training mode
        loss_arr = []
        for x, y in train_loader:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)  # move data to device (cpu/cuda)
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update model with optimizer
            loss_arr.append(mse_loss.item())
        print(f"epoch: {epoch}/{config['n_epochs']} , loss: {np.mean(loss_arr)}")
        epoch += 1

    print('Finished training after {} epochs'.format(epoch))


def find_min_distance_word_vector(cur_i, vector, embeddings, vocabulary):
    def calc_distance(v1, v2):
        # 计算欧式距离
        distance = 0
        for i in range(len(v1)):
            distance += sqrt(pow(v1[i] - v2[i], 2))
        return distance

    min_distance = None
    min_i = -1
    for i, word in enumerate(vocabulary):
        if cur_i != i:
            distance = calc_distance(vector, embeddings[i].tolist())
            if min_distance is None or min_distance > distance:
                min_distance = distance
                min_i = i
    return min_i


if __name__ == '__main__':
    # data_path = 'D:\\Projects\\NLP_Projects\\nlp\\model_exp_word2vec_10_20\\process_txt\\ICD-10.txt'
    # data_path = 'D:\\Projects\\NLP_Projects\\nlp\\model_exp_word2vec_10_20\\process_txt\\train-text.txt'
        # 指定要读取的YAML文件路径
    config = 'config.yaml'

    # 使用yaml库加载YAML文件为Python字典
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_path=config['data_path']
    same_seed(config['seed'])

    data_set, vocabulary, index_dict = get_data_set(data_path, config['window_width'], config['window_step'],
                                                    config['negative_sample_num'])
    train_loader = DataLoader(data_set, config['batch_size'], shuffle=True, drop_last=False, pin_memory=True)

    model = DNN(len(vocabulary), config['embedding_dim']).to(config['device'])

    train(model, train_loader, config)

    # 训练完，看看embeddings，展示部分词的词向量，并找到离它最近的词的词向量
    embeddings = torch.t(model.embedding.weight)
    for i in range(10):
        print('%-50s%s' % (f"{vocabulary[i]} 的词向量为 :", str(embeddings[i].tolist())))
        min_i = find_min_distance_word_vector(i, embeddings[i].tolist(), embeddings, vocabulary)
        print('%-45s%s' % (
            f"离 {vocabulary[i]} 最近的词为 {vocabulary[min_i]} , 它的词向量为 :", str(embeddings[min_i].tolist())))
        print('-' * 200)
