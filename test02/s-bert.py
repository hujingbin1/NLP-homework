# 模型预测
from sentence_transformers import SentenceTransformer
import scipy.spatial
import csv

# csv转字符串列表
def read_csv_file(file_path):
    with open(file_path, 'r',encoding='UTF-8') as file:
        csv_reader = csv.reader(file)
        data = [','.join(row) for row in csv_reader]
    return data

# 统计CSV文件的行数
def count_csv_rows(file_path):
    with open(file_path, 'r',encoding='UTF-8') as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)
    return row_count

#Micro-F1得分计算函数
def Micro_f1(m,n,k):
   P=k/n
   R=k/m
   if P+R==0:
      return 0
   else:
    F1=2*P*R/(P+R)
    return F1

# 读取CSV文件并将每一行作为字符串元素生成字符串列表
#评估集的输入文本
file_path1 = r'..\data\data_final_exp_10_29\dev_data_F1\dev-text.csv'
text1 = read_csv_file(file_path1)
number1 = count_csv_rows( file_path1)
#评估集的待选文本
file_path2 = r'..\data\data_final_exp_10_29\dev_data_F1\ICD-10.csv'
text2 = read_csv_file(file_path2)
number2 = count_csv_rows(file_path2)
#评估集的答案文本
file_path3 = r'..\data\data_final_exp_10_29\dev_data_F1\dev-normalized.csv'
text3= read_csv_file(file_path3)
number3 = count_csv_rows(file_path3)

tuple_text1=[tuple(s.split('##')) for s in text1]
tuple_text3=[tuple(s.split('##')) for s in text3]

embedder = SentenceTransformer('test_output')
corpus = text2
corpus_embeddings = embedder.encode(corpus)
# 待查询的句子
queries = text1
query_embeddings = embedder.encode(queries)
# 对于每个句子，使用余弦相似度查询最接近的n个句子
closest_n = 5
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
  print("Number:",tag)
  print("Query:", query)
  print("Result:Top 1 most similar sentences in corpus:")
  for idx, distance in results[0:closest_n]:
    print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
    if ((corpus[idx].strip() in tuple_text3[tag])  and flag == 0):
       flag=1
       right=right+1
  tag=tag+1
  print("Micro-F1得分：",Micro_f1(number1,tag,right))
  print("======================")

