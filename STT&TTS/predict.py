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
from collections import Counter
import json
import jieba
import re
import openpyxl
from  clean import clean

class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list

class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        return out.last_hidden_state[:, 0]  # [batch, 768]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
POOLING = 'cls'  
# 预训练模型目录
model_path = '..\\test01\\chinese_roberta_wwm_ext_pytorch'
model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
model.load_state_dict(torch.load("..\\test01\\simcse_sup.pt"))
model.to(DEVICE)
print("ok")

from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
MAXLEN = 64
def encode(model,text):
    model.eval()
    source = tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
    source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
    source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
    source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
    return source_pred 


def predict(s):
    text = clean(s)    
    icd_df = pd.read_excel(
    '..\\origin_data\\ICD-10.xlsx',
    header=None, 
    names=['icd_code', 'name'])
    icd_name = icd_df["name"]
    where_res=np.where(icd_name==text)
    if len(where_res[0]) == 0 :
        icd_name = np.append(icd_name,text)
    document_name = [list(jieba.cut(t)) for t in icd_name]
    bm25_model = BM25_Model(document_name)
    info = []
    with torch.no_grad():
        query = list(jieba.cut(text))
        scores = bm25_model.get_documents_score(query)
        results = zip(range(len(scores)), scores)
        results = sorted(results, key=lambda x: x[1],reverse=True)
        query_emb = encode(model,text)
        scores_sim = []
        for num,score in results[0:400]:
            sen1 = encode(model,icd_name[num])
            sim = F.cosine_similarity(query_emb, sen1, dim=-1)
            scores_sim.append(sim)
        results_sim = zip(range(len(scores_sim)), scores_sim)
        results_sim = sorted(results_sim, key=lambda x: x[1],reverse=True)
        se = ""
        for num , score in results_sim[0:2]:
                 se = se +"##"+ str(icd_name[results[num][0]])
        return se