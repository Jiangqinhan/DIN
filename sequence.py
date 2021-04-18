import numpy as np
import tensorflow as tf
from utils import reduce_max
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers import LSTM, Lambda, Layer
from utils import reduce_max, reduce_mean, reduce_sum, div


class SequencePoolingLayer(Layer):
    '''
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
        如果 supports_masking=True 输入为seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``而不是一个列表

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

        - **supports_masking**:If True,the input need to support masking.


        产生的效果描述:输入embedding (batch_size,input_length,embedding_size) length=(batch_size) maxlen=input_len
        最终的mask (batch_size,input_len,embedding_size) embedding*mask
        最后一维全为1 或 0
    '''

    def __init__(self, mode, supports_masking=False, **kwargs):
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        super(SequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking
        self.eps = tf.constant(1e-8, tf.float32)

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            seq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)

        else:
            seq_embed_list, user_behavior_length = seq_value_len_list
            '''
                a = tf.sequence_mask([1, 2, 3], 5)
                [[ True False False False False]
                [ True  True False False False]
                [ True  True  True False False]]
                最大长度为5 表示一共5个 第一个参数是长度list，所以第一个样本 取值长度为1
                tf.sequence_mask([[1, 2], [3, 4]]) 
                [[[ True False False False]
                [ True  True False False]]

                [[ True  True  True False]
                [ True  True  True  True]]]  
                实际在call时输出的维度维(?,1,max_len)
            '''
            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))
        embedding_size = seq_embed_list.shape[-1]
        '''
        a = tf.constant([[1,2],[3,4]],name='a')   
        b = tf.tile(a,[2,3])
        [[1 2 1 2 1 2]
        [3 4 3 4 3 4]
        [1 2 1 2 1 2]
        [3 4 3 4 3 4]]
        '''
        mask = tf.tile(mask, [1, 1, embedding_size])
        if self.mode == 'max':
            hist = seq_embed_list - (1 - mask) * 1e9
            hist = reduce_max(hist, 1, keep_dims=True)
        hist = reduce_sum(seq_embed_list * mask, axis=1, keep_dims=False)

        if self.mode == 'mean':
            # 防止出现0作为分母
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)
        hist = tf.expand_dims(hist, axis=1)
        return hist


    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def get_config(self):
        config = config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSequenceLayer(Layer):
    """The WeightedSequenceLayer is used to apply weight score on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of three  tensor [seq_value,seq_len,seq_weight]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

        - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``
        或者 输入为
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``
        supports_masking=True
      Output shape
        - 3D tensor with shape: ``(batch_size, T, embedding_size)``.

      Arguments
        - **weight_normalization**: bool.Whether normalize the weight score before applying to sequence.

        - **supports_masking**:If True,the input need to support masking.
    """
    def __init__(self,weight_normalization=True,supports_masking=False,**kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.supports_masking=supports_masking
        self.weight_normalization=weight_normalization

    def build(self,input_shape):
        if not self.supports_masking:
            self.seq_len_max=int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(input_shape)

    def call(self,input_list,mask=None,**kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            embedding_input,weight_input=input_list
            #只把embedding的mask拿出来
            mask=tf.expand_dims(mask[0],axis=2)
        else:
            embedding_input,length_input,weight_input=input_list
            mask=tf.sequence_mask(length_input,self.seq_len_max,dtype=tf.bool)
            mask=tf.transpose(mask,(0,2,1))

        embedding_size=embedding_input.shape[-1]
        if self.weight_normalization:




    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        if self.supports_masking:
            return mask[0]
        else:
            return None

    def get_config(self):
        config = {'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

