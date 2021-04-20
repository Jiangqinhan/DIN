from tensorflow.python.keras.layers import Embedding,Lambda
from tensorflow.python.keras.regularizers import l2
from collections import defaultdict
from utils import Hash
from itertools import chain
from sequence import SequencePoolingLayer,WeightedSequenceLayer
def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    '''
    :param sparse_feature_columns:
    :param varlen_sparse_feature_columns:
    :param seed:
    :param l2_reg:
    :param prefix:
    :param seq_mask_zero:
    :return: dict {embedding_name:EmbeddingLayer} shape(batchsize,input_len,embedding_dim)
    '''
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim, embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg)
                        , name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb
    '''
    第一个if语句是由于     varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        引起的
    可变长的特征 要用mask
    l2()产生一个l1l2类对象 实现 正则化操作
    '''
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    '''
    将需要embedding的特征都挑出来 调用create_embedding_dict
    :param feature_columns: list[SparseFeat/DenseFear/VarLenSparseFeat]
    :param l2_reg:
    :param seed:
    :param prefix:
    :param seq_mask_zero: 0是否用于填充
    :return:dict {embedding_name:EmbeddingLayer} shape(batchsize,input_len,embedding_dim) 从这里开始有了mask
    '''
    from feature_column import SparseFeat,VarLenSparseFeat
    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_embedding_dict=create_embedding_dict(sparse_feature_columns,varlen_sparse_feature_columns,seed,l2_reg,prefix=prefix + 'sparse',seq_mask_zero=seq_mask_zero)

    return sparse_embedding_dict

def embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns,return_feat_list=(),mask_feat_list=(),to_list=False):
    '''

    Args:
        sparse_embedding_dict: {embedding_name:embeddingLayer}
        sparse_input_dict: {feature_name:Input}有变长特征
        sparse_feature_columns: [SparseFeat/VarLenSparseFeat]根据名字把Input和Embedding匹配起来
        return_feat_list:实际想要的feat 如果这个列表的长度为0 就是想要所有的feat
        mask_feat_list:
        to_list:确定返回格式

    Returns:{group_name: [embeddingLayer(Input)]}把embedding和Input对应上
            或者list[embeddingLayer(Input)]


    '''
    group_embedding_dict=defaultdict(list)
    for feat in sparse_feature_columns:
        feature_name=feat.name
        embedding_name=feat.embedding_name
        if len(return_feat_list)==0 or feature_name in return_feat_list:
            if feat.use_hash:
                lookup_idx=Hash(feat.vocabulary_size,mask_zero=(feature_name in mask_feat_list))(sparse_input_dict[feature_name])
            else:
                lookup_idx=sparse_input_dict[feature_name]
            group_embedding_dict[feat.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        '''
        chain.from_iterable 把几个list的东西合并到一个迭代器里
        '''
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

def get_dense_input(features,feature_columns):
    '''

    :param features: {feature_name:Input}
    :param feature_columns: [DenseFeat]
    :return:[Input]
    '''
    from feature_column import DenseFeat
    dense_feature_list=list(filter(lambda x:isinstance(x,DenseFeat),feature_columns))
    dense_input_list=[]
    for fc in dense_feature_list:
        if fc.transform_fc is None:
            dense_input_list.append(features[fc.name])
        else:
            tran_result=Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(tran_result)
        return dense_input_list

def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    '''
    与embeddinglookup效果相似但是只用于varlenSparseFeat且返回为dict,原因是还要进行池化需要名字去对应其他变量
    Args:
        embedding_dict:
        sequence_input_dict:
        varlen_sparse_feature_columns:

    Returns:dict{feature_name:embedding_layer}
    '''
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict

def get_varlen_pool_list(embedding_dict,features,varlen_sparse_feature_columns,to_list=False):
    pool_vec_list=defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_nanme=fc.name
        feature_length_name=fc.length_name
        combiner=fc.combiner
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input=WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_nanme],features[fc.weight_name]])

            else:
                seq_input=embedding_dict[feature_nanme]
            vec=SequencePoolingLayer(combiner,supports_masking=False)([seq_input,features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input=WeightedSequenceLayer(weight_normalization=fc.weight_norm,supports_masking=True)(
                embedding_dict[feature_nanme],features[fc.weight_name])
            else:
                seq_input=embedding_dict[feature_nanme]
            vec=SequencePoolingLayer(combiner,supports_masking=True)(seq_input)
        pool_vec_list[fc.group_name].append(vec)
    if to_list:
        return list(chain.from_iterable(pool_vec_list.values()))
    return pool_vec_list



