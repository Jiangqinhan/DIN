import  tensorflow as tf

class NoMask(tf.keras.layers.Layer):
    '''
    主要是为了保证支持mask
    '''
    def __int__(self,**kwargs):
        super(NoMask, self).__int__(**kwargs)

    def build(self, input_shape):
        super(NoMask, self).build(input_shape)

    def call(self,x,mask=None,**kwargs):
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

def concat_func(inputs,axis=-1,mask=False):
    '''

    :param inputs: list
    :param axis: 默认为最后一维 例如 em
    :param mask:
    :return:
    '''
    if not mask:
        return list(map(NoMask(),inputs))
    if len(inputs)==1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)



class Hash(tf.keras.layers.Layer):
    pass
