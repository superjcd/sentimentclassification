'''
  Evaluate model performence
'''
import pandas as pd
import os
import torch
import argparse
from utils import load_data_to_csv,  transform_data
import model
from dataset import TEXT, LABEL


# 确保存在test.csv
if not os.path.exists('data/test.csv'):
    load_data_to_csv(flag='test')


parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='birnn',choices=['textcnn', 'birnn'], help='choose one model name for trainng')
parser.add_argument('-lmd', '--load-model-dir', default= None, help='path for loadding model, default:None' )
args = parser.parse_args()

# 获取模型名称
net = getattr(model, args.model_name)()
net.load_state_dict(torch.load(args.load_model_dir))


def evaluate(model, df):
    result = {'correct':0, 'wrong':0}
    df_len = df.shape[0]
    for i in range(df_len):
        record = df.loc[i, :].to_dict()
        data, label = transform_data(record, TEXT, LABEL)
        score = model(data)
        if score.argmax(dim=1) == label:
            result['correct'] += 1
        else:
            result['wrong'] += 1
    print(f"Classification Accuracy of Model({model.__class__.__name__})is {result['correct']/df_len} ")



if __name__ == '__main__':
    test_data = pd.read_csv('data/test.csv')
    evaluate(model=net, df=test_data)











