import tensorflow as tf
from feature_column import build_input_features, SparseFeat, VarLenSparseFeat, DenseFeat
from inputs import create_embedding_matrix, embedding_lookup, get_dense_input, get_varlen_pool_list, \
    varlen_embedding_lookup
from utils import concat_func
from core import DNN, PredictionLayer
from utils import NoMask, combine_dnn_input, reduce_sum, reduce_mean
from sequence import AttentionSequencePoolingLayer, DynamicGRU
from tensorflow.python.keras.layers import Dense,Permute


def auxiliary_loss(h_state, click_sequence, no_click_sequence, mask, stag):
    '''
    实现时应该注意
    :param h_state:[batch_size,input_length,embedding_size]
    :param click_sequence:
    :param no_click_sequence:
    :param mask: 负样本的长度要与正样本对应,也是变长的
    :param stag: 类似name
    :return:shape()
    '''
    hist_length = h_state.get_shape()[1]
    mask = tf.sequence_mask(mask, hist_length)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask)
    postive_mul = h_state * click_sequence
    neg_mul = h_state * no_click_sequence
    postive_inner_product = reduce_sum(postive_mul, axis=2) * mask
    neg_inner_product = reduce_sum(neg_mul, axis=2) * mask
    postive = tf.log(tf.sigmoid(postive_inner_product))
    neg = tf.log(1.0 - tf.sigmoid(neg_inner_product))
    # 不指定维数就将所有的元素相加求平均
    return reduce_mean(postive + neg)

def auxiliary_loss1(h_states, click_seq, noclick_seq, mask, stag=None):
    #:param h_states:
    #:param click_seq:
    #:param noclick_seq: #[B,T-1,E]
    #:param mask:#[B,1]
    #:param stag:
    #:return:
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    mask = tf.sequence_mask(mask, hist_len)
    mask = mask[:, 0, :]

    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)

    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    auxiliary_nn = DNN([100, 50, 1], activation='sigmoid')

    click_prop_ = auxiliary_nn(click_input_, stag=stag)[:, :, 0]

    noclick_prop_ = auxiliary_nn(noclick_input_, stag=stag)[
                    :, :, 0]  # [B,T-1]

    try:
        click_loss_ = - tf.reshape(tf.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
    except:
        click_loss_ = - tf.reshape(tf.compat.v1.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
    try:
        noclick_loss_ = - \
                            tf.reshape(tf.log(1.0 - noclick_prop_),
                                       [-1, tf.shape(noclick_seq)[1]]) * mask
    except:
        noclick_loss_ = - \
                            tf.reshape(tf.compat.v1.log(1.0 - noclick_prop_),
                                       [-1, tf.shape(noclick_seq)[1]]) * mask

    loss_ = reduce_mean(click_loss_ + noclick_loss_)

    return loss_

def interest_extraction_and_evolution(concat_behaviour, deep_input_item, user_behaviour_length, gru_type, use_neg=False,
                                      neg_concat_behaviour=None, att_hidden_size=(64, 16), att_activation='sigmoid',
                                      att_weight_normalization=False):
    '''

    :param concat_behaviour:
    :param deep_input_item:
    :param user_behaviour_length:
    :param gru_type:
    :param use_neg:
    :param neg_concat_behaviour:
    :param att_hidden_size:
    :param att_activation:
    :param att_weight_normalization:
    extraction统一使用grue
    在evolution部分 如果使用gru 则 再进行一个gru层然后使用sequence_attention进行计算出最终的h
    其他情况下 调用相应的gru 并且将att_score传入其中
    :return: final_state [batch_size,1,embedding_size]   loss
    '''
    if gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
        raise ValueError("gru_type error")
    aux_loss = None
    embedding_size = None
    # extraction
    rnn_output = DynamicGRU(num_units=embedding_size, return_sequence=True, name='gru1')(
        (concat_behaviour, user_behaviour_length))
    if use_neg:
        # 特别注意 要把对应关系搞清楚 在论文中h_t 对应 e_(t+1)
        aux_loss = auxiliary_loss(rnn_output[:, :-1, :], concat_behaviour[:, 1:, :], neg_concat_behaviour[:, 1:, :],
                                  user_behaviour_length, stag='gru')
    # evolution
    if gru_type == 'GRU':
        rnn_output_2 = DynamicGRU(embedding_size, return_sequence=True, name='gru2')(
            [rnn_output, user_behaviour_length])
        hist = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                             weight_normalization=att_weight_normalization, return_score=False)(
            [deep_input_item, rnn_output, user_behaviour_length])
    else:
        att_score = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                                  weight_normalization=att_weight_normalization, return_score=True)(
            [deep_input_item, rnn_output, user_behaviour_length])
        if gru_type == 'AIGRU':
            rnn_output = Permute([2,1])(att_score) * rnn_output
            hist = DynamicGRU(embedding_size,gru_type=gru_type, return_sequence=False, name='gru2')([rnn_output, user_behaviour_length])
        else:
            hist = DynamicGRU(embedding_size,gru_type=gru_type, return_sequence=False, name='gru2')([rnn_output, user_behaviour_length,Permute([2,1])(att_score)])
    return hist, aux_loss


def DIEN(dnn_feature_columns, history_feature_list,
         gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(200, 80),
         dnn_activation='relu', att_hidden_units=(64, 16), att_activation="dice", att_weight_normalization=True,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, task='binary'):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param gru_type: str,can be GRU AIGRU AUGRU AGRU
    :param use_negsampling: bool, whether or not use negative sampling
    :param alpha: float ,weight of auxiliary_loss
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    features = build_input_features(dnn_feature_columns)
    user_behaviour_length = features['seq_length']

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    neg_history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: 'hist_' + x, history_feature_list))
    neg_history_fc_names = list(map(lambda x: 'neg_' + x, history_fc_names))
    # history的特征都是不定长的,将变长的特征进行分类
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        elif feature_name in neg_history_fc_names:
            neg_history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    input_list = list(features.values())
    # 处理相应的embedding
    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix='',
                                             seq_mask_zero=False)
    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list, to_list=True)
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)
    sequence_emb_dict = varlen_embedding_lookup(embedding_dict, features, varlen_sparse_feature_columns)
    sequence_emb_list = get_varlen_pool_list(sequence_emb_dict, features, sparse_varlen_feature_columns, to_list=True)
    #
    keys_emb = concat_func(keys_emb_list)
    query_emb = concat_func(query_emb_list)
    dnn_input_emb_list += sequence_emb_list
    deep_input_emb = concat_func(dnn_input_emb_list)

    if use_negsampling:

        neg_uiseq_embed_list = embedding_lookup(embedding_dict, features, neg_history_feature_columns,
                                                neg_history_fc_names, to_list=True)

        neg_concat_behavior = concat_func(neg_uiseq_embed_list)

    else:
        neg_concat_behavior = None
    hist, aux_loss = interest_extraction_and_evolution(keys_emb, query_emb, user_behaviour_length, gru_type=gru_type,
                                                       use_neg=use_negsampling,
                                                       neg_concat_behaviour=neg_concat_behavior,
                                                       att_hidden_size=att_hidden_units, att_activation=att_activation,
                                                       att_weight_normalization=att_weight_normalization)

    deep_input_emb = concat_func([deep_input_emb, hist], axis=-1)
    dnn_input = combine_dnn_input([deep_input_emb], dense_value_list)
    out_put = DNN(hidden_units=dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                  use_bn=use_bn, seed=seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(out_put)
    final_logit = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=input_list, outputs=final_logit)
    if use_negsampling:
        model.add_loss(-alpha * aux_loss)
    try:
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
    except:
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    return model
