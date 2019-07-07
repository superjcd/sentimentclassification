'''
  Evaluate model performence
'''
import pandas as pd
import os
import torch
from utils import load_data_to_csv,  transform_data
from model import textcnn, birnn

from dataset import TEXT, LABEL


# 确保存在test.csv
if not os.path.exists('data/test.csv'):
    load_data_to_csv(flag='test')



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
    net = textcnn()
    net.load_state_dict(torch.load('models_storage/model_cnn.pt'))
    test_data = pd.read_csv('data/test.csv')
    evaluate(model=net, df=test_data)











