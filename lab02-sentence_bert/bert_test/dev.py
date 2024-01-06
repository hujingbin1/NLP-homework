from sklearn.metrics import f1_score

# 实际标签
true_labels = [0, 1, 2, 0, 1, 2, 0, 2, 2]
# 预测标签
predicted_labels = [0, 2, 1, 0, 1, 1, 0, 2, 2]

# 计算 Micro-F1 得分
micro_f1 = f1_score(true_labels, predicted_labels, average='micro')

print("Micro-F1 Score:", micro_f1)