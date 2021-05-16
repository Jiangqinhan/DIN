from utils import to_df,build_map
import pickle
import numpy as np
import random
from test_function import test_DIN,test_DeepFM,test_dien
import time
import tensorflow as tf
from utils import reduce_sum

if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    '''
    #test_DIN()
    start_time=time.time()
    test_DeepFM()
    print('time cost {}'.format(time.time()-start_time))


    a=tf.constant([0.1,0.2,0.3,0.4,0.5,0.6],shape=[2,3],dtype=tf.float32)
    b=tf.constant([2,6,3,5,8,1],shape=[2,3],dtype=tf.float32)
    res = a*b
    res=reduce_sum(res,axis=1)

    res=tf.sigmoid(res)
    with tf.Session() as sess:
        print("res_value:", sess.run(res))
    print(1/(np.exp(-2.3)+1))
    '''
    test_DIN()



