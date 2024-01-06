# py39pt20



![image](https://ask.qcloudimg.com/http-save/yehe-1599485/c72d432ec559263489578305913e4a80.png)


语义匹配：

* ## 百度的SimNet框架，有监督学习，词袋模型

python=3.6 tensorflow=1.2.0

```shell
pip install tensorflow-gpu==1.2  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install configparser
pip install utils
```

* ## 自己搭建的word2vec框架(词向量叠加得到句子特征向量【这块我想加注意力机制，不仅仅是简单叠加，着重关注名词、动词的权重】)，无监督学习，效果不一定好，但是训练速度快

python=3.9 torch=2.0.1 

**注意：pip换源清华源，关梯子，会下载很快，如果清华镜像源没有，则换阿里等别的源，否则挂梯子。**

**最好用conda下，这样容易对包统一管理，pip下容易引起包版本不同形成冲突。**

```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas
conda install openpyxl
conda install jieba
```

* ## sentence+bert，预训练模型

  * python 3.9

* ## SimCSE（Chinese）

python=3.9 torch=2.0.1 

```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install jsonlines -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install loguru -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```



* ## test开源（word2vec+lstm）[有监督？]

```shell
pip install utils
pip install jieba 
```

* DSSM 词义相似度匹配

* BioBert
* umlsbert







# 语音识别/语音合成：（调用百度的api）

* **百度的SimNet框架，有监督学习，词袋模型**

python=3.6 tenserflow=1.2.0

```shell
pip install pyaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install baidu-aip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install chardet -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pygame -i https://pypi.tuna.tsinghua.edu.cn/simple
#注意使用时关闭代理
```

* **sentence+bert，预训练模型**

python=3.9 torch=2.0.1

```shell
pip install pyaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install baidu-aip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install chardet -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pygame -i https://pypi.tuna.tsinghua.edu.cn/simple
#注意使用时关闭代理
```

f1分数测试

```shell
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

jupyter notebook

```shell
python -m ipykernel install --user --name py39pt20
```







