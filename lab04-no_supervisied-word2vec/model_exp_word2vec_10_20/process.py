import csv

# 读取CSV文件并删除反斜杠
# with open(r'D:\Projects\NLP_Projects\nlp\data_pre_process_ICD-10_10_20\ICD-10.csv', 'r', newline='', encoding='utf-8') as csv_file:
#     reader = csv.reader(csv_file)
#     data = [[cell.replace('/', '') for cell in row] for row in reader]

# # 保存为TXT文件
# with open(r'.\process_txt\ICD-10.txt', 'w', encoding='utf-8') as txt_file:
#     for row in data:
#         txt_file.write('\t'.join(row) + '\n')

# 读取CSV文件并删除反斜杠
with open(r'D:\Projects\NLP_Projects\nlp\data_exp_10_17\train-text.csv', 'r', newline='', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)
    data = [[cell.replace('/', '') for cell in row] for row in reader]

# 保存为TXT文件
with open(r'process_txt/train-text.txt', 'w', encoding='utf-8') as txt_file:
    for row in data:
        txt_file.write('\t'.join(row) + '\n')