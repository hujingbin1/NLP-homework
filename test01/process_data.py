import json
import pandas as pd
import numpy as np 
import re
import openpyxl
import jsonlines
from clean import clean 

#我在这里开始处理数据
icd_df = pd.read_excel(
    '..\\origin_data\\ICD-10.xlsx',
    header=None, 
    names=['icd_code', 'name'])

out_data = [] 
with open('..\\origin_data\\train.json', encoding='utf-8') as df:
    results = json.load(df)
    for result in results :
        text = clean(result.get('text'))
        norm_lists = clean(result.get('normalized_result')).split('##')
        for norm_list in norm_lists :
            origin = text
            entailment = norm_list
            #通过ICD-10标准数据集中随机挑选的文本作为负样例来进行label的设定
            contradiction = list(np.array(icd_df["name"].sample(1)))[0]
            data = {'origin': origin, 'entailment': entailment, 'contradiction': contradiction} 
            out_data.append(data)
with jsonlines.open('.\\ICD', 'w') as writer:
    writer.write_all(out_data)

#将训练集拆分为训练集和验证集
out_data_train = out_data[:8000]
out_data_dev = out_data[8000:10948]

with jsonlines.open('.\\ICD_train.txt', 'w') as writer:
    writer.write_all(out_data_train)

#借鉴了ark-nlp在CHIP2021临床属于标准化方法中的数据集处理文件
ainfo = []
for data in out_data_dev:
    data1 = data["origin"] + "||" + data["entailment"] + "||" + "1"
    ainfo.append(data1)
    data1 = data["origin"] + "||" + data["contradiction"] + "||" + "0"
    ainfo.append(data1)
with jsonlines.open('.\\ICD_dev.txt', 'w') as writer:
    writer.write_all(ainfo)