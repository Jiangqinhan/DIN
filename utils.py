import tensorflow as tf
from tensorflow.python.keras.layers import Flatten
import pickle
import pandas as pd


class NoMask(tf.keras.layers.Layer):
    '''
    主要是为了保证支持mask
    '''

    def __int__(self, **kwargs):
        super(NoMask, self).__int__(**kwargs)

    def build(self, input_shape):
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask=None):
        return None


'''
    这些函数主要是是为了兼容性
'''


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


def concat_func(inputs, axis=-1, mask=False):
    '''

    :param inputs: list
    :param axis: 默认为最后一维 例如 em
    :param mask:
    :return:
    '''
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def combine_dnn_input(sparse_embedding_list, dense_value_list):
    '''
    针对多维的特征,将其Flatten
    :param sparse_embedding_list:
    :param dense_value_list:
    :return:
    '''
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


class Hash(tf.keras.layers.Layer):
    """
        hash the input to [0,num_buckets)
        if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
        """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):

        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df


def build_map(df, col_name):
    '''
    制作一个map,键名为列名 值为序列数字,用于将文字数据转化为数字数据
    :param df:
    :param col_name:
    :return: key,map
    '''
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


class Linear(tf.keras.layers.Layer):
    '''
    mode 0 表示只有稀疏特征
         1 表示只有稠密特征
         2 表示两种特征都有
    '''

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):
        self.l2_reg = l2_reg
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1,2")
        self.mode = mode
        self.use_bias = use_bias
        self.seed = seed
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True
                                        )
        if self.mode == 1:
            if self.mode == 1:
                self.kernel = self.add_weight(
                    'linear_kernel',
                    shape=[int(input_shape[-1]), 1],
                    initializer=tf.keras.initializers.glorot_normal(self.seed),
                    regularizer=tf.keras.regularizers.l2(self.l2_reg),
                    trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight('linear_kernel',
                                          shape=[int(input_shape[1][-1]), 1],
                                          initializer=tf.keras.initializers.glorot_normal(self.seed),
                                          regularizer=tf.keras.regularizers.l2(self.l2_reg)
                                          )
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Add(tf.keras.layers.Layer):
    def __int__(self,**kwargs):
        super(Add, self).__int__(**kwargs)

    def build(self, input_shape):
        super(Add, self).build(input_shape)

    def call(self,inputs,*kwargs):
        if not isinstance(input,list):
            return inputs
        if len(inputs)==1:
            return inputs[0]
        if len(inputs)==0:
            return tf.constant([[0.0]])

        return tf.keras.layers.add(inputs)

def add_func(inputs):
    return  Add()(inputs)

def log(input):
    try:
        return tf.log(input)
    except AttributeError:
        return tf.math.log(input)
