import numpy as np
import tensorflow as tf
from utils import reduce_max
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers import LSTM, Lambda, Layer
from utils import reduce_max, reduce_mean, reduce_sum, div, softmax
from core import LocalActivationUnit
from Layer_utils import AGRUCell, AUCRUCell
from rnn import dynamic_rnn


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

    def __init__(self, weight_normalization=True, supports_masking=False, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking
        self.weight_normalization = weight_normalization

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(input_shape)

    def call(self, input_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            embedding_input, weight_input = input_list
            # 只把embedding的mask拿出来
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            embedding_input, length_input, weight_input = input_list
            mask = tf.sequence_mask(length_input, self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = embedding_input.shape[-1]
        if self.weight_normalization:
            paddings = tf.ones_like(weight_input) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(weight_input)
        weight_input = tf.where(mask, weight_input, paddings)

        if self.weight_normalization:
            weight_input = softmax(weight_input, dim=1)
        weight_input = tf.tile(weight_input, [1, 1, embedding_size])
        return tf.multiply(embedding_input, weight_input)

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


class AttentionSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in DIN.

          Input shape
            - A list of three tensor: [query,keys,keys_length]

            - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

            - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

            - keys_length is a 2D tensor with shape: ``(batch_size, 1)``
            或者为supports_masking==True 时
             - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

            - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``


          Output shape
            - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

          Arguments
            - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

            - **att_activation**: Activation function to use in attention net.

            - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

            - **supports_masking**:If True,the input need to support masking.

          References
            - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 3:
                raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
                                 'on a list of 3 inputs')
            if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
                raise ValueError(
                    "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                        len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))
            if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
                raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                                 'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                                 'Got different shapes: %s' % (input_shape))
        else:
            pass
        super(AttentionSequencePoolingLayer, self).build(input_shape)
        self.local_att = LocalActivationUnit(self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0,
                                             use_bn=False, seed=1024)

    def call(self, inputs, mask=None, training=None, **kwargs):
        # query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            queries, keys = inputs
            key_masking = tf.expand_dims(mask[-1], axis=1)
        else:
            queries, keys, keys_length = inputs
            hist_len = keys.shape[1]
            key_masking = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)
        outputs = tf.transpose(attention_score, (0, 2, 1))
        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)
        outputs = tf.where(key_masking, outputs, paddings)
        if self.weight_normalization:
            outputs = softmax(outputs)
        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicGRU(Layer):
    '''

    :param num_units: 通常用于表示输出的维数state 和 output的维数
    :param gru_type: "GRU", "AIGRU", "AGRU", "AUGRU"
    :param return_sequence: 返回output list  或者是 最终的state
    :param kwargs:
    :return:
    '''

    def __init__(self, num_units=None, return_sequence=True, gru_type="GRU", **kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        self.gru_type = gru_type
        super(DynamicGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
         self.num_units 默认为 embedding_size
        :param input_shape:
        :return:
        '''
        if self.num_units is None:
            self.num_units = input_shape[0][-1]
        if self.gru_type == 'AGRU':
            self.gru_cell = AGRUCell(self.num_units)
        elif self.gru_type == 'AUGRU':
            self.gru_cell = AUCRUCell(self.num_units)
        else:
            self.gru_cell = tf.nn.rnn_cell.GRUCell(self.num_units)

        super(DynamicGRU, self).build(input_shape)

    def call(self, input_list):
        '''

        :param input_list: shape[batch_size,field_size,embedding_size], sequence_length用于表示迭代次数,att_score,[batch_size,field_size,1]
        :return:output [batch_size,field_size,embedding_size] 或者finial_state: batch_size,embedding_size
        '''
        if self.gru_type == 'AUGRU' or self.gru_type == 'AGRU':
            rnn_input, sequence_length, att_score = input_list
        else:
            rnn_input, sequence_length = input_list
            att_score = None
        #squeeze是dynamic_rnn的要求
        rnn_output, final_state = dynamic_rnn(self.gru_cell, rnn_input,att_score,
                                              tf.squeeze(sequence_length,) ,dtype=tf.float32,scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(final_state,axis=1)

    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None, 1, rnn_input_shape[-1])

    def get_config(self, ):
        config = {'num_units': self.num_units, 'gru_type': self.gru_type, 'return_sequence': self.return_sequence}
        base_config = super(DynamicGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
