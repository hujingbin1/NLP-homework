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

def predict(s):
    embedder = SentenceTransformer('..\\test02\\test_output')
    #评估集的待选文本
    file_path = r'..\data\data_final_exp_10_29\dev_data_F1\ICD-10.csv'
    text = read_csv_file(file_path)
    corpus = text
    corpus_embeddings = embedder.encode(corpus) 
    # 待查询的句子
    q=[]
    queries = q+[s]
    query_embeddings = embedder.encode(queries)
    # 对于每个句子，使用余弦相似度查询最接近的n个句子
    closest_n = 1
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        # 按照距离逆序
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        for idx, distance in results[0:closest_n]:
            return (str(corpus[idx].strip()))
            