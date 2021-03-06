from feature_column import build_input_features, SparseFeat, VarLenSparseFeat, DenseFeat
from inputs import create_embedding_matrix,embedding_lookup,get_dense_input,get_varlen_pool_list,varlen_embedding_lookup
from utils import concat_func
import tensorflow as tf
from core import DNN,PredictionLayer
from utils import NoMask,combine_dnn_input
from sequence import AttentionSequencePoolingLayer
def DIN(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
        task='binary'):
    '''

    :param dnn_feature_columns: 用于模型的各种特征相关的信息 list[SparseFeat/DenseFear/VarLenSparseFeat]
    :param history_feature_list: list,to indicate  sequence sparse field [str] 里面的内容为特征名字
        要求特征名字 带hist_的是历史数据 而不带 hist是所有商品的数据
    :param dnn_use_bn:
    :param dnn_hidden_units:
    :param dnn_activation:
    用于生成权重的dnn
    :param att_hidden_size:
    :param att_activation:
    :param att_weight_normalization:
    :param l2_reg_dnn:
    :param l2_reg_embedding:
    :param dnn_dropout:
    :param seed:
    :param task:
    :return: A keras model instance
    '''
    features = build_input_features(dnn_feature_columns)
    # filter 返回迭代器对象
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    # 可变长度的变量被分为两类
    history_feature_columns = []
    sparse_varlen_columns = []
    his_fc_names = list("hist_" + x for x in history_feature_list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in his_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_columns.append(fc)
    inputs_list = list(features.values())
    embedding_dict=create_embedding_matrix(dnn_feature_columns,l2_reg_embedding,seed,prefix='')
    #这两个要参与atention的计算
    '''
        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
    '''
    query_emb_list=embedding_lookup(embedding_dict,features,sparse_feature_columns,history_feature_list,history_feature_list,to_list=True)
    key_emb_list=embedding_lookup(embedding_dict,features,history_feature_columns,his_fc_names,his_fc_names, to_list=True)
    # query和dnn_input_emb共用embedding
    dnn_input_embedding_list=embedding_lookup(embedding_dict,features,sparse_feature_columns,mask_feat_list=history_feature_list, to_list=True)
    dense_value_list=get_dense_input(features,dense_feature_columns)

    sequence_embedding_dict=varlen_embedding_lookup(embedding_dict,features,varlen_sparse_feature_columns)

    sequence_embed_list = get_varlen_pool_list(sequence_embedding_dict, features, varlen_sparse_feature_columns,
                                                  to_list=True)
    #除了需要attention以外的所有离散特征 包括VarLenSparseFeat
    dnn_input_embedding_list += sequence_embed_list
    keys_emb=concat_func(key_emb_list,mask=True)
    deep_input_emb=concat_func(dnn_input_embedding_list)
    query_emb=concat_func(query_emb_list,mask=True)
    hist=AttentionSequencePoolingLayer(att_hidden_size,att_activation,weight_normalization=att_weight_normalization,supports_masking=True)([
        query_emb,keys_emb
    ])
    deep_input_emb=tf.keras.layers.Concatenate()([NoMask()(deep_input_emb),hist])
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    dnn_input = combine_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model