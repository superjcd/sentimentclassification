'''
 将数据从本地存储到mysql数据库
'''

import pymysql
import sys
sys.path.append('..')
from configs import BasicConfigs


bc = BasicConfigs()

train = open('raw_data/train.txt', encoding='utf-8')
test = open('raw_data/test.txt', encoding='utf-8')

def get_data(line):
    '''
    提取原始语料中的关键信息
    :param line: 每行文字
    :return: data:文本， label:标签
    '''
    infos = line.split(',')
    #print(infos)
    label = int(infos[1])
    data = ''.join(infos[2:])
    return [data, label]


def insert_db_many(datas, table=bc.table_name):
    conn = pymysql.connect(**bc.db_connection)
    cursor = conn.cursor()
    query = '''INSERT INTO {}(data, label, flag) VALUES (%s, %s, %s);'''.format(table)
    try:
        cursor.executemany(query, datas)
    except Exception as e:
        raise e
    print('Record inserted to Database')
    conn.commit()
    conn.close()


def load_data(file, flag, every=10000):
    '''
     将数据load到database
    :param every:
    :return:
    '''
    datas = []
    for i, line in enumerate(file):
        record = get_data(line)
        record.append(flag)
        print(record)
        datas.append(record)
        if (i+1) % every == 0:
            insert_db_many(datas)
            datas = []
    # 将剩余的数据也load起来
    insert_db_many(datas)


if __name__ == '__main__':
    load_data(file=train, flag='train')
    load_data(file=test, flag='test')
    train.close()
    test.close()






