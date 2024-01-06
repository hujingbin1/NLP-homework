import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # 读取CSV文件
    with open(csv_file_path, 'r', encoding='UTF-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        # 将CSV数据转换为字典列表
        data = [row for row in csv_reader]

    # 写入JSON文件
    with open(json_file_path, 'w', encoding='UTF-8') as json_file:
        # 将字典列表写入JSON文件
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# 指定CSV文件和JSON文件的路径
# csv_file_path = r'..\data_final_exp_10_29\dev_data_F1\test-final.csv'
# json_file_path = r'..\data_final_exp_10_29\result\test-final.json'
csv_file_path = r'..\data_final_exp_10_29\dev_data_F1\test.csv'
json_file_path = r'..\data_final_exp_10_29\result\test.json'

# 调用函数将CSV文件转换为JSON文件
csv_to_json(csv_file_path, json_file_path)