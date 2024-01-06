import re
import csv
import json
import pandas as pd
import numpy as np
import jsonlines
import openpyxl
from  clean import clean

#读取json文件
with open(r'..\origin_data\test.json', 'r', encoding='utf-8') as jsonfile:
    json_string = json.load(jsonfile)    


#调用清洗函数对text和normalied进行数据清洗
text = []
normalized = []
for i in range(len(json_string)):
    json_string[i]['text']=clean(json_string[i]['text'])
    text.append(json_string[i]['text'])
    json_string[i]['normalized_result']=clean(json_string[i]['normalized_result'])
    normalized.append(json_string[i]['normalized_result'])

#结果保存csv格式
csv_file = open(r'..\data_final_exp_10_29\dev_data_F1\test.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
# csv_file = open(r'..\data_final_exp_10_29\dev_data_F1\dev-normalized.csv', 'w', newline='', encoding='utf-8')
# writer = csv.writer(csv_file)
# writer.writerow(['Text', 'Normalized Result'])
writer.writerow(['Text'])

#把结果以CSV格式存储
for i in range(len(text)):
    writer.writerow([text[i]])
    # writer.writerow([normalized[i]])
    # writer.writerow([text[i],normalized[i]])
  
csv_file.close()



