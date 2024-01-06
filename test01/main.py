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
from data_load import *
from model import *
from eval import *
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
#模型训练函数，其下游任务为评估验证集准确率
def train(model, train_dl, dev_dl, optimizer) -> None:
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dl)
            model.train()
            if best < corrcoef:
                early_stop_batch = 0
                best = corrcoef
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue
            early_stop_batch += 1
            if early_stop_batch == 100:
                logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                logger.info(f"train use sample number: {(batch_idx - 10) * BATCH_SIZE}")
                return 


logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
tokenizer = BertTokenizer.from_pretrained(model_path)

#加载数据
train_data = load_data('train',TRAIN)
random.shuffle(train_data)                        
dev_data = load_data('dev',DEV) 
train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=BATCH_SIZE)
print("data ok ")

#加载模型
model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
# 训练
best = 0
for epoch in range(EPOCHS):
    logger.info(f'epoch: {epoch}')
    train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')

#验证
model.load_state_dict(torch.load(SAVE_PATH))
dev_corrcoef = eval(model, dev_dataloader)
logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')