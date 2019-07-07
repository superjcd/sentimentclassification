### snetimentclassification ###
###     Author: JCD         ###
###     Date: 2019/07/01    ###
###############################
import os
from configs import BasicConfigs

# 确保词向量缓存目录存在
bc = BasicConfigs()
cache = bc.cach
if not os.path.exists(cache):
    os.mkdir(cache)

