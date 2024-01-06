# 模型预测
from sentence_transformers import SentenceTransformer
import scipy.spatial
import re
import csv
import json
import pandas as pd
import numpy as np
#r为需要删除的字符
r = "\\【.*?】+|\\《.*?》+|[.!/_,$&%^*()<>+\"'?@|:~{}\\[\\]\\s-]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
#数据清洗，去除异常值与奇怪字符
def convert(text):
    halfwidth_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"  # 半角符号
    fullwidth_chars = "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"  # 全角符号
    #全角转半角字符
    for i in range(len(halfwidth_chars)):
        text = text.replace(fullwidth_chars[i],halfwidth_chars[i])
    #删除r中的字符
    text= re.sub(r, '', text)
    #去除异常值
    cleaned_text = re.sub(r'\u0004+', '', text)
    cleaned_text = re.sub(r'\ufeff+', '', cleaned_text )
    #把text中的分隔符也替换为'##'
    cleaned_text= cleaned_text.replace(";", "##")
    return cleaned_text


# 统计CSV文件的行数
def count_csv_rows(file_path):
    with open(file_path, 'r',encoding='UTF-8') as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)
    return row_count


def tostring():
    file_path='D:\\Projects\\NLP_Projects\\nlp\\dev.json'
    #读取json文件
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        json_string = json.load(jsonfile)
        number = count_csv_rows(file_path)
    #调用清洗函数对text和normalied进行数据清洗
    text = []
    normalied = []
    for i in range(len(json_string)):
        json_string[i]['text']=convert(json_string[i]['text'])
        text.append(json_string[i]['text'])
        json_string[i]['normalized_result'] = convert(json_string[i]['normalized_result'])
        normalied.append(json_string[i]['normalized_result'])

    return text,normalied,number

#读入excel文件以字符列表保存
def read_excel():
    df = pd.read_excel('D:\\Projects\\NLP_Projects\\nlp\\ICD-10.xlsx', usecols=[1])
    column = df.iloc[:, 0].tolist()
    strings_list = [str(cell) for cell in column]
    return strings_list

#调用清洗函数进行数据清洗
def cleanlist(text):
    for i in range(len(text)):
        text[i]=convert(text[i])
        text.append(text[i])
    return text

#Micro-F1得分计算函数
def Micro_f1(m,n,k):
   P=k/n
   R=k/m
   if P+R==0:
      return 0
   else:
    F1=2*P*R/(P+R)
    return F1

text,noramlied,number=tostring()
tuple_normalied=[tuple(s.split('##')) for s in noramlied]

#答案列表
answer_list=read_excel()
answer_list=cleanlist(answer_list)
#选择预训练的语言模型
embedder = SentenceTransformer('bert-base-chinese')
# 备选的文本集合 ICD-10
corpus = answer_list
corpus_embeddings = embedder.encode(corpus)
# 待查询的句子 text
queries = text
query_embeddings = embedder.encode(queries)
# # 正确答案 normalized 用于计算F1得分
# corrects = noramlied
# corrects_embeddings=embedder.encode(corrects)
# 对于每个句子，使用余弦相似度查询最接近的n个句子
closest_n = 1
#记录循环次数，表明第几个数据
tag=0
#记录正确预测的个数
right=0
for query, query_embedding in zip(queries, query_embeddings):
  flag=0
  distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
  # 按照距离逆序
  results = zip(range(len(distances)), distances)
  results = sorted(results, key=lambda x: x[1])
  print("======================")
  print("Query:", query)
  print("Result:Top 1 most similar sentences in corpus:")
  for idx, distance in results[0:closest_n]:
      print(corpus[idx].strip(), "(Score: %.4f)" % (1 - distance))
      print(tuple_normalied[tag])
      if corpus[idx].strip() in tuple_normalied[tag] and flag == 0:
          flag = 1
          right = right + 1
  tag = tag + 1
  print("Micro-F1得分：", Micro_f1(number, tag, right))