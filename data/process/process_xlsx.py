import pandas as pd
import re
import jieba
import csv
import json
from  clean import clean
#调用清洗函数进行数据清洗
def clean_text(text):
    for i in range(len(text)):
        text[i]=clean(text[i])
        text.append(text[i])
    return text

#结果保存csv格式
def save(text):
    csv_file = open('..\data_final_exp_10_29\dev_data_F1\ICD-10.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    #结果以CSV格式存储
    for i in range(len(text)):
        writer.writerow([text[i]])

    csv_file.close()

if __name__ == "__main__":
    text=read_excel()
    text=clean_text(text)
    save(text)


