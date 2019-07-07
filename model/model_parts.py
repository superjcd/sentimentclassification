import torch.nn.functional as F
from torch import nn

class GlobalMaxPool1d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, X):
        return F.max_pool1d(X, kernel_size=X.shape[2])  # 在seqence进行global



class Flatten(nn.Module):
    '''
      用来展开除batch以外的tensor
    '''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)