import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

# 准备训练数据
sentences = [["I", "love", "coding"],
             ["Word2Vec", "is", "awesome"],
             ["Machine", "learning", "is", "fascinating"]]

# 转换数据为字符串形式
sentences = [" ".join(sentence) for sentence in sentences]

# 加载预训练的词向量模型
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# 获取句子的向量表示
sentence_vectors = embed(sentences)

# 构建Word2Vec模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(sentences), 100, input_length=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100)
])

# 编译模型
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam())

# 训练模型
model.fit(sentence_vectors, sentence_vectors, epochs=10)

# 获取词向量
word = "Word2Vec"
word_vector = model.layers[0].get_weights()[0][sentences.index(word)]
print(word_vector)