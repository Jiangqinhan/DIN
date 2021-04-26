from collections import namedtuple, OrderedDict
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.layers import Input
from copy import copy
from inputs import embedding_lookup, create_embedding_dict, create_embedding_matrix, varlen_embedding_lookup, \
    get_dense_input, get_varlen_pool_list
from inputs import mergeDict
from itertools import chain
from utils import concat_func, Linear

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(
    namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embeddings_initializer',
                              'embedding_name',
                              'group_name', 'trainable'])):
    '''
    不允许给对象增加新的属性
    group_name 决定是否参与某些操作,例如在deepfm中,决定是否参加fm的内积运算
    '''
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fc'])):
    '''
    默认维度是1
    数据预处理是None
    '''
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32', transform_fc=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fc)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    '''
    lengthname 实际的数据中带有填充 length_name 对应了每个样本实际数据长度的列表
    combiner 变长数据转化为定长数据的方式 例如mean
    maxlen :用于为Input的维度赋值
    VarLenSparseFeat实际数据的内容 是某种稀疏特征 因此 包含该特征相关的信息
    weight 相关信息 用于产生weight层
    '''
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


def build_input_features(feature_columns, prefix=''):
    '''
    根据输入信息生成相应的Input层,这里用字典存储是非常重要的feature_columns中的特征名字可能会重复,但是input不能重复
    :param feature_columns: list[SparseFeat/DenseFear/VarLenSparseFeat]
    :param prefix:
    :return:dict {name:Input}
    '''
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    '''
    根据feature_columns 构造embedding_layer 并且与INput进行匹配  对 varlen sparse feat 进行池化处理,抽取dense feat的input
    :param features:
    :param feature_columns:
    :param l2_reg:
    :param seed:
    :param prefix:
    :param seq_mask_zero:
    :param support_dense:
    :param support_group:
    :return: dict{group_name:list[embedding_layer]}或者 list[embedding-layer],list[Input]
    '''
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                  seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    seq_embedding_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pool_list(seq_embedding_dict, features,
                                                              varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list


def get_linear_logit(features, feature_columns, units=1, use_bias=False, seed=1024, prefix='linear', l2_reg=0):
    '''
    实现线性回归,对于varlensparsefeat 用SequencePoolingLayer处理 ,这里对离散变量进行1维的embedding就相当于产生了wi,所以在线性层中只要加起来就行了
    :return:
    '''
    linear_feature_columns = copy(feature_columns)
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1,
                                                                           embeddings_initializer=Zeros())
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(
                sparsefeat=linear_feature_columns[i].sparsefeat._replace(embedding_dim=1,
                                                                         embeddings_initializer=Zeros())
            )
    linear_emb_list = [
        input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix + str(i))[0] for i in
        range(units)]
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)
    linear_logit_list = []
    for i in range(units):
        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg=l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])
        elif len(linear_emb_list[i])>0:
            sparse_input=concat_func(linear_emb_list[i])
            linear_logit=Linear(l2_reg=l2_reg,mode=0,use_bias=use_bias,seed=seed)(sparse_input)
        elif len(dense_input_list)>0:
            dense_input=concat_func(dense_input_list)
            linear_logit=Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:
            raise NotImplementedError
        linear_logit_list.append(linear_logit)
    return concat_func(linear_logit_list)



