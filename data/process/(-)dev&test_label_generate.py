import json
import pandas as pd
import numpy as np
import re
import jsonlines
import csv
import openpyxl
from  clean import clean


icd_df = pd.read_excel(
    '..\\origin_data\\ICD-10.xlsx',
    header=None,
    names=['icd_code', 'name'])

out_data = []
with open('..\\origin_data\\dev.json', encoding='utf-8') as df:
    results = json.load(df)
    for result in results :
        text = clean(result.get('text'))
        norm_lists = clean(result.get('normalized_result')).split('##')
        for norm_list in norm_lists :
            origin = text
            entailment = norm_list
            contradiction = list(np.array(icd_df["name"].sample(1)))[0]
            data = {'origin': origin, 'entailment': entailment, 'contradiction': contradiction}
            out_data.append(data)

out_data_dev = out_data[:2001]
out_data_test = out_data[2001:3542]

with jsonlines.open('..\\data_final_exp_10_29\\label_exp_dev_10_29\\ICD_dev.txt', 'w') as writer:
    writer.write_all(out_data_dev)

with jsonlines.open('..\\data_final_exp_10_29\\label_exp_test_10_29\\ICD_test.txt', 'w') as writer:
    writer.write_all(out_data_test)

##############dev
ainfo = []
for data in out_data_dev:
    data1 = data["origin"] + "||" + data["entailment"] + "||" + "1"
    ainfo.append(data1)
    data1 = data["origin"] + "||" + data["contradiction"] + "||" + "0"
    ainfo.append(data1)

with jsonlines.open('..\\data_final_exp_10_29\\label_exp_dev_10_29\\ICD_dev_label.txt', 'w') as writer:
    writer.write_all(ainfo)

with open('..\\data_final_exp_10_29\\label_exp_dev_10_29\\ICD_dev_label.txt', 'r',encoding='utf-8') as file:
    lines = file.readlines()

#加入表头
# 在第一行前插入表头
header=['Text', 'Normalized Result','label']
lines.insert(0, ','.join(header))

#去掉""
lines = [line.strip().replace('"', '') for line in lines]
# 去掉每一行的换行符
lines = [line.strip() for line in lines]


data = [line.strip().split('||') for line in lines]

with open('..\\data_final_exp_10_29\\label_exp_dev_10_29\\ICD_dev_label.csv', 'w', newline='',encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)
############test
ainfo = []
for data in out_data_test:
    data1 = data["origin"] + "||" + data["entailment"] + "||" + "1"
    ainfo.append(data1)
    data1 = data["origin"] + "||" + data["contradiction"] + "||" + "0"
    ainfo.append(data1)

with jsonlines.open('..\\data_final_exp_10_29\\label_exp_test_10_29\\ICD_test_label.txt', 'w') as writer:
    writer.write_all(ainfo)

with open('..\\data_final_exp_10_29\\label_exp_test_10_29\\ICD_test_label.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

#加入表头
# 在第一行前插入表头
header=['Text', 'Normalized Result','label']
lines.insert(0, ','.join(header))

#去掉""
lines = [line.strip().replace('"', '') for line in lines]

# 去掉每一行的换行符
lines = [line.strip() for line in lines]

data = [line.strip().split('||') for line in lines]

with open('..\\data_final_exp_10_29\\label_exp_test_10_29\\ICD_test_label.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)