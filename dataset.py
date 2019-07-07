'''
  构建Voaabulary
  构建文本迭代器
'''

import jieba
from configs import BasicConfigs
from torchtext.vocab import Vectors
from torchtext.data import Field, TabularDataset, BucketIterator
from utils import chi_tokenizer


bc = BasicConfigs()

# 定义Field
TEXT = Field(tokenize=chi_tokenizer) # 在这例可以添加很多有用的参数， 比如pa_token，unknowntoken,stopwords
LABEL = Field(eos_token=None, pad_token=None, unk_token=None)

# 定义字段与FIELD之间读配对
fields = [('data', TEXT), ('label',LABEL)]

# 注意skip_header
train, val = TabularDataset.splits(path='data',train='train.csv',
                                   validation='val.csv',
                                   format='csv',
                                   fields=fields,
                                   skip_header=True)

# train, val = TabularDataset().splits(path='./data', train='train.csv', validation='val.csv',
#                                    format='csv', fields=fields, skip_header=True)

#  构建从本地加载的词向量
vectors = Vectors(name=bc.embedding_loc, cache=bc.cach)

# 构建vocabulary
TEXT.build_vocab(train, val, vectors=vectors)
LABEL.build_vocab(train, val, vectors=vectors)

# print(LABEL.vocab.stoi['0']) # '1':2, '0':3

train_iter = BucketIterator(train, batch_size=bc.batch_size, \
sort_key=lambda x: len(x.data), sort_within_batch=True, shuffle=True)

val_iter = BucketIterator(val, batch_size=bc.batch_size, \
sort_key=lambda x: len(x.data), sort_within_batch=True, shuffle=True)

vocab_size = TEXT.vocab.vectors.shape


if __name__ == '__main__':
    print(TEXT.vocab.vectors.shape)
    # print(LABEL.vocab.stoi['1']) # 0
    # print(LABEL.vocab.stoi['0']) # 1







