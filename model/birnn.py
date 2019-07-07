import torch
from torch import nn
from dataset import TEXT, vocab_size
from configs import BasicConfigs


bc = BasicConfigs()

class BiRNN(nn.Module):
    def __init__(self, num_hiddens=bc.num_hiddens, num_layers=bc.num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=vocab_size[1],
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs