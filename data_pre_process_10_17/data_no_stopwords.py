# 导入分词库
import jieba
import re
import csv
import json

# # 过滤不了\\ \ 中文（）还有————
# r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'#用户也可以在此进行自定义过滤字符
# # 者中规则也过滤不完全
# r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
# # \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
# r3 =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
# # 去掉括号和括号内的所有内容
# # r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
# # r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+\"'?@|:~{}#‘’]+|[——！\\\，。=？、：“”￥……（）《》【】]"

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

#jieba分词
def cuttest(test_sent):
    result = jieba.cut(test_sent)
    return " / ".join(result)

# 打开去除停用词txt文件
with open('D:\\Projects\\NLP_Projects\\nlp\\pre_process\\stop_words\\stop_words.txt', 'r', encoding='GBK') as file:
    # 读取文件内容
    content = file.read()

# 将内容按行分割为列表
lines = content.split('\n')

# 去除空行
lines = [line.strip() for line in lines if line.strip()]


#读取json文件
with open('D:\\Projects\\NLP_Projects\\nlp\\train.json', 'r', encoding='utf-8') as jsonfile:
    json_string = json.load(jsonfile)
# with open('D:\\Projects\\NLP_Projects\\nlp\\dev.json', 'r', encoding='utf-8') as jsonfile:
#     json_string = json.load(jsonfile)
# with open('D:\\Projects\\NLP_Projects\\nlp\\test.json', 'r', encoding='utf-8') as jsonfile:
#     json_string = json.load(jsonfile)    


#调用清洗函数对text和normalied进行数据清洗
text = []
normalized = []
for i in range(len(json_string)):
    json_string[i]['text']=convert(json_string[i]['text'])
    text.append(json_string[i]['text'])
    json_string[i]['normalized_result']=convert(json_string[i]['normalized_result'])
    normalized.append(json_string[i]['normalized_result'])

#结果保存csv格式
csv_file = open('train.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('dev.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('test.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('train-text.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('train-normalized.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('dev-text.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('dev-normalized.csv', 'w', newline='', encoding='utf-8')
# csv_file = open('test-text.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
# writer.writerow(['Text'])
# writer.writerow(['Normalized Result'])
writer.writerow(['Text', 'Normalized Result'])

#调用jieba分词模块进行分词并把结果以CSV格式存储，同时进行停用词处理
for i in range(len(text)):
    text[i]=[word for word in text[i] if word not in lines]
    text_str = ''.join(text[i])
    text_cut = cuttest(text_str)

    normalized[i]=[word for word in normalized[i] if word not in lines]
    normalized_str=''.join(normalized[i])
    normalized_cut=cuttest(normalized_str)

    # writer.writerow([text_cut])
    # writer.writerow([normalized_cut])
    writer.writerow([text_cut,normalized_cut])
  
csv_file.close()
