from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
import pandas as pd
#设置参数
model_name = 'bert-base-chinese'
train_batch_size =48 #32 #32 #16
num_epochs =15 #4 #10 #4
model_save_path = 'test_output'
logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])
#使用Huggingface/transformers模型（如BERT）将标记映射到嵌入向量。
word_embedding_model = models.Transformer(model_name)
#应用均值池化（mean pooling）以获得一个固定大小的句子向量。
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
  pooling_mode_mean_tokens=True,
  pooling_mode_cls_token=False,
  pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_samples = []
dev_samples = []
test_samples = []
#加载数据
def load(path):
  df = pd.read_csv(path)
  samples = []
  for idx,item in df.iterrows():
    samples.append(InputExample(texts=[item['Text'], item['Normalized Result']],label=float(item['label'])))
  return samples
#读入csv文件
train_samples = load(r'..\data\data_final_exp_10_29\label_exp_train_10_29\ICD_train_label.csv')
test_samples = load(r'..\data\data_final_exp_10_29\label_exp_test_10_29\ICD_test_label.csv')
dev_samples = load(r'..\data\data_final_exp_10_29\label_exp_dev_10_29\ICD_dev_label.csv')
#加载数据
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
#10%的训练数据用于warm-up
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
# 模型训练
model.fit(train_objectives=[(train_dataloader, train_loss)],
  evaluator=evaluator,
  epochs=num_epochs,
  evaluation_steps=1000,
  warmup_steps=warmup_steps,
  output_path=model_save_path)
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)