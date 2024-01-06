import random
import time
from typing import List
import pandas as pd
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
from data_load import *
from model import *
import yaml
config = 'config.yaml'
# 使用yaml库加载YAML文件为Python字典
with open(config, 'r',encoding='utf-8') as f:
    config = yaml.safe_load(f)
# 基本参数
EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
LR = config['LR']
MAXLEN = config['MAXLEN']
POOLING = config['POOLING']  

# 预训练模型目录
model_path =config['model_path']

# 微调后参数存放位置
SAVE_PATH = config['SAVE_PATH']

# 数据位置
TRAIN = config['TRAIN']
DEV = config['DEV']

DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#评估函数，数据来自data_process构建的验证集
def eval(model, dataloader) -> float:
    model.eval()
    label_array = np.array([])
    acc = 0 
    num = 5896
    thresholds = [0.6,0.65,0.7 , 0.75, 0.8,0.85, 0.9,0.95 ]
    for threshold in thresholds:
        acc_now = 0
        with torch.no_grad():
            for source, target, label in dataloader:
                # source        [batch, 1, seq_len] -> [batch, seq_len]
                source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
                source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
                source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
                source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
                # target        [batch, 1, seq_len] -> [batch, seq_len]
                target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
                target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
                target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
                target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
                # concat
                sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
        # corrcoef  
                sim_numpy = sim.cpu().numpy()
                sim_n = np.array([])
                for s,l in zip(sim_numpy,label):
                    if s >= threshold :
                        sim_n = np.append(sim_n,1)
                    else :
                        sim_n = np.append(sim_n,0)
                acc_now = acc_now + np.count_nonzero(sim_n==label.cpu().numpy())
            acc = max (acc , acc_now)
    print(acc/num)
    return acc/num