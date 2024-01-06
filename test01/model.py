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
#模型以及损失函数，这里采用SimCSE有监督版本
class SimcseModel(nn.Module):
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        return out.last_hidden_state[:, 0] 
                  
            
def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)