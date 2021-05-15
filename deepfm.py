from itertools import chain
from feature_column import DEFAULT_GROUP_NAME,build_input_features,input_from_feature_columns,get_linear_logit
from utils import combine_dnn_input,add_func,concat_func
from core import DNN,PredictionLayer
import tensorflow as tf
from interaction import FM

def DeepFM(linear_feature_columns,dnn_feature_columns,fm_group=[DEFAULT_GROUP_NAME],dnn_hidden_units=(128,128),
           l2_reg_linear=0.00001,l2_reg_embedding=0.00001,l2_dnn_reg=0,seed=1024,dnn_dropout_rate=0,dnn_activation='relu',dnn_use_bn=False,task='binary'):
    """Instantiates the DeepFM Network architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
            只是用来产生FM的线性部分
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
            这里的mebedding为FM和dnn共用
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    features=build_input_features(linear_feature_columns+dnn_feature_columns)
    #用于最后输入
    input_list=list(features.values())
    #实现dnn部分
    group_embedding_dict,dense_value_list=input_from_feature_columns(features,dnn_feature_columns,l2_reg_embedding,seed,support_group=True)
    #group_embedding_dict是字典 需要chain处理
    dnn_input=combine_dnn_input(list(chain.from_iterable(group_embedding_dict.values())),dense_value_list)
    dnn_output=DNN(dnn_hidden_units,dnn_activation,l2_dnn_reg,dnn_dropout_rate,dnn_use_bn,seed=seed)(dnn_input)
    dnn_logit=tf.keras.layers.Dense(1,use_bias=False,kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    #实现fm部分
    fm_logit=None
    linear_logit=get_linear_logit(features,linear_feature_columns,seed=seed,prefix='linear',l2_reg=l2_reg_linear)
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    #求和
    linear_logit=tf.expand_dims(linear_logit,axis=0)
    dnn_logit=tf.expand_dims(dnn_logit,axis=0)
    final_logit=add_func([linear_logit,fm_logit,dnn_logit])

    output=PredictionLayer(task)(final_logit)
    model=tf.keras.models.Model(inputs=input_list,outputs=output)
    return model










