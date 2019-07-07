import pandas as pd
import pymysql
import jieba
import torch
from sklearn.model_selection import train_test_split
from configs import BasicConfigs


# 加载
jieba.load_userdict('dictionary/机构_学校.lex')

bc = BasicConfigs()

def load_data_to_csv(flag):
    conn = pymysql.connect(**bc.db_connection)
    if flag == 'train':
        data = pd.read_sql('select data,label from weibo_sentiment where flag="train"', conn)
        train_data, val_data = train_test_split(data, test_size=0.1)
        train_data.to_csv("data/train.csv", index=False)
        val_data.to_csv("data/val.csv", index=False)
    elif flag == 'test':
        test_data = pd.read_sql('select data,label from weibo_sentiment where flag="test"', conn)
        test_data.to_csv('data/test.csv')
    else:
        raise ValueError('flag must be in ("test", "train")')


# 定义一个tokenizer
def chi_tokenizer(sentence):
    return [word for word in jieba.cut(sentence)]


def transform_data(record, TEXT, LABEL):
    # [[2,8,9]] => [[7], [8], [9]]
    if not isinstance(record, dict):
        raise ValueError('Make sure data is dict')
    tokens = chi_tokenizer(record['data'])
    res = []
    for token in tokens:
        res.append(TEXT.vocab.stoi[token])
    data = torch.tensor(res).unsqueeze(1)
    if 'label' in list(record):
        label = torch.tensor(LABEL.vocab.stoi[record['label']])
    else:
        label = None
    return data, label





if __name__ == '__main__':
    from dataset import TEXT, LABEL
    record = {'data':'真是太开心了', 'label':1}
    data, label = transform_data(record, TEXT, LABEL)
    print(data, label)




