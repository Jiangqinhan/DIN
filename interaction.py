import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import batch_dot
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform, TruncatedNormal)
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers import utils

from activation import activation_layer
from utils import concat_func, reduce_sum, softmax, reduce_mean

class FM(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
         without linear term and bias.
         直接得出 sigma<vi,vj> i,j 为所有非0的特征
         计算方法: 在field维度先计算和的平方在计算平方的和相减,然后*0.5 最后在embedding维度求和

          Input shape
            - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

          Output shape
            - 2D tensor with shape: ``(batch_size, 1)``.

          References
            - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """
    def __int__(self,**kwargs):
        super(FM, self).__int__(**kwargs)

    def build(self, input_shape):
        if len(input_shape)!=3:
            raise ValueError("Unexpected inputs dimension %d,\
                             expect to be 3 dimension" % (len(input_shape)))
        super(FM, self).build(input_shape)

    def call(self,inputs,**kwargs ):
        if K.ndim(inputs)!=3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value=inputs
        square_sum=tf.square(reduce_sum(concated_embeds_value,axis=1,keep_dims=True))
        sum_square=reduce_sum(concated_embeds_value * concated_embeds_value,axis=1,keep_dims=True)
        cross_term=square_sum-sum_square
        cross_term=0.5*reduce_sum(cross_term,axis=2,keep_dims=False)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None,1)





