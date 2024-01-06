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

tokenizer = BertTokenizer.from_pretrained(model_path)

#数据读取以及装入到dataset
def load_data(name: str, path: str) -> List:
    def load_train_data(path):        
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]
        
    def load_dev_data(path):
        with open(path, 'r', encoding='utf8') as f:            
            return [(line.split("||")[0], line.split("||")[1], line.split("||")[2]) for line in f] 
    if name == 'train':
        return load_train_data(path)    
    return load_dev_data(path)
    

class TrainDataset(Dataset):
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index]) 

class TestDataset(Dataset):
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2].replace('"\n',''))
    