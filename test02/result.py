from sentence_transformers import SentenceTransformer
import scipy.spatial
import csv
import pandas as pd

# csv转字符串列表
def read_csv_file(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        csv_reader = csv.reader(file)
        data = [','.join(row) for row in csv_reader]
    return data

# 统计CSV文件的行数
def count_csv_rows(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        csv_reader = csv.reader(file)
        row_count = sum(1 for row in csv_reader)
    return row_count

# 读取CSV文件并将每一行作为字符串元素生成字符串列表
# 评估集的输入文本
file_path1 = r'..\data\data_final_exp_10_29\dev_data_F1\test.csv'
text1 = read_csv_file(file_path1)
number1 = count_csv_rows(file_path1)

# 评估集的待选文本
file_path2 = r'..\data\data_final_exp_10_29\dev_data_F1\ICD-10.csv'
text2 = read_csv_file(file_path2)
number2 = count_csv_rows(file_path2)

tuple_text1 = [tuple(s.split('##')) for s in text1]

embedder = SentenceTransformer('test_output')
corpus = text2
corpus_embeddings = embedder.encode(corpus)

# 待查询的句子
queries = text1
query_embeddings = embedder.encode(queries)

# 对于每个句子，使用余弦相似度查询最接近的n个句子
closest_n = 1
result = []
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    # 按照距离逆序
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1 - distance))
        result.append(corpus[idx].strip())
#结果保存csv文件中
df = pd.read_csv(r'..\data\data_final_exp_10_29\dev_data_F1\test.csv')
df.insert(1, 'Normalized Result', result[1:])
df.to_csv(r'..\data\data_final_exp_10_29\dev_data_F1\test.csv', index=False)

