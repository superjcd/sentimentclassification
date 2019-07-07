import torch
import os
import argparse
from torch import optim
from torch import nn
import model
from train import train
from dataset import train_iter, val_iter
from utils import load_data_to_csv
from configs import  BasicConfigs


# 确保有'train.csv' 文件在指定目录下
if not os.path.exists('data/train.csv'):
    load_data_to_csv(flag='train')

bc = BasicConfigs()

# 获取参数
parser = argparse.ArgumentParser()
parser.add_argument('--compute-val', action='store_true', help='compute validation accuracy or not, default:None')
parser.add_argument('--epoches', default=20, type=int, help='num of epoches for trainning loop, default:20')
parser.add_argument('-lmd', '--load-model-dir', default= None, help='path for loadding model, default:None' )
parser.add_argument('-smd', '--save-model-dir', default=None, help='models_storage/model_cnn.pt, defaul:None')
parser.add_argument('--model-name', default='birnn',choices=['textcnn', 'birnn'], help='choose one model name for trainng')
args = parser.parse_args()

# 获取模型名称
net = getattr(model, args.model_name)()
device = bc.device

optimizer = optim.Adam(net.parameters(), lr=bc.lr)
loss_func = nn.CrossEntropyLoss()


if __name__ == '__main__':
    train(net=net, optimizer=optimizer, loss_func=loss_func, \
          train_iter=train_iter, val_iter=val_iter,
          compute_val=args.compute_val, device=device, epoches=args.epoches,
          load_model_dir=args.load_model_dir, save_model_dir=args.save_model_dir)
