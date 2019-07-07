# 微博情感分类实验
  该项目是使用pytorch的基于TexctCNN， 以及双向循环神经网络的文本分类实验：对微博的短博文进行情感二分类。当然， 该项目也可以应用于多分类问题。
  
## 项目的整体结构 
```
├── configs.py         配置文件
├── data               原始数据
├── dataset.py         数据格式转换
├── dictionary         用户自定义词典
├── embeddings         词向量
├── evaluation.py      模型在测试集上测试效果
├── main.py            主程序
├── models.py          文本分类模型
├── models_storage     模型保存目录
├── train.py           训练相关函数
└── utils.py           工具函数
```

## 使用方法
（1） 准备好数据
   data目录下准备好训练数据集：train.csv、val.csv(如果要在训练中实时获取validation accuracy的话)
 数据要保证有以下：data(文本字段)、 label(分类标签， 该例分为0和1， 0代表消极， 1代表积极)。本项目使用的数据来源于[这里](https://github.com/dengxiuqi/weibo2018)。  
 **重要提醒**： 由于使用torchtext对label进行了tokenize时，会把原始数据当作文本类别， 本例中模型输出的0对应的是积极、1对应的是消极（也就是和原标签的意义相反）
   
（2） 下载词向量  
  从网上下载预训练好的词向量， 比如FastText词向量, 你可从[该处](https://fasttext.cc/docs/en/crawl-vectors.html)
下载一个300维的中文词向量， 然后将解压出来的txt词向量文本文件放在embeddings/fasttext/目录下（在配置文件标明相应的词向量地址）

（3） 修改配置文件
  如果data目录下没有trian.csv（test.cv同理）， 程序会默认从配置文件所定义的数据库连接中获取数据，
需要保证自定义的数据表有以下字段：
  - data, 文本数据
  - label, 分类标签
  - flag，值为 'train' 或 'test' 二者之一
  
（4） 运行及保存模型  
  在
   
   
（5） 测试模型  
在evaluation.py加载好模型,以及读取完测试集数据后， 可以运行： 

```python
 python evaluation.py
```

## 模型的评估结果
|模型名称 |准确率 |
|---|---|
|BiRNN（双向循环神经网络（lstm））| 70.4%|
|TextCNN| 71.2%|

