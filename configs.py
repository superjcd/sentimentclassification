import torch

class BasicConfigs():
    # parameters for db connection
    db_type = 'mysql'
    db_connection = {'host':'127.0.0.1',
                     'user':'root',
                     'password':'****',
                     'db':'corpus'}  # 数据库链接， 如果在data目录下已经有了train.csv就不需要考虑
    table_name = 'weibo_sentiment'   # 同上
    # parmeters for wordvector
    embedding_loc = 'embeddings/fasttext/cc.zh.300.vec'
    cach = '.vector_cache'   # 词向量的缓存位置
    # parameters for overall model training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.005          # learning rate
    dropout_rate = 0.5
    train_embedding = True
    batch_size = 64
    # parameter for textcnn
    kernel_sizes = [3, 4, 5]  # 3个 conv1d的size
    num_channels = [100, 100, 100]
    # paramaeter for birnn
    num_hiddens = 100
    num_layers = 2




class CustomConfigs(BasicConfigs):
    '''
      这里， 可以自定义参数
    :return:
    '''
    pass


if __name__ == '__main__':
    bc = BasicConfigs()
    print(bc.db_connection)