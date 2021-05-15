from feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
import numpy as np
from din import DIN
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import metrics
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from deepfm import DeepFM
from tensorflow.python.keras import backend as K
from dien import DIEN
#from deepctr.models import DeepFM
#from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


folder = r"D:\Amozon_data_set"


def get_xy_fd(method=None):
    '''

    :param method: 目前支持的测试 DeepFM和DIN
    :return:
    '''
    feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat(
        'gender', 2, embedding_dim=8), SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=8), DenseFeat('pay_score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=8, embedding_name='cate_id'), maxlen=4,
                         length_name="seq_length")]
    # Notice: History behavior sequence feature name must start with "hist_".
    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    seq_length = np.array([3, 3, 2])  # the actual length of the behavior sequence

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': seq_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    if method=='DeepFM':
        behavior_feature_list=[SparseFeat('user', 3, embedding_dim=8), SparseFeat(
        'gender', 2, embedding_dim=4), SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=8), DenseFeat('pay_score', 1)]
    return x, y, feature_columns, behavior_feature_list

def get_config(method=None):
    with open(folder + r"/dataset.pkl", 'rb') as f:
        train_set=pickle.load(f)
        test_set=pickle.load(f)
        cate_list=pickle.load(f)
        (user_count, item_count, cate_count)=pickle.load(f)
    #label和其他相关的数据
    y=np.array([sample[-1] for sample in train_set])
    reviewerID=np.array([sample[0] for sample in train_set])
    asin=np.array([sample[-2] for sample in train_set])
    seq_length=np.array([len(sample[1])for sample in train_set])
    max_len=np.max(seq_length)
    hist_asin=[sample[1] for sample in train_set]
    hist_asin=pad_sequences(hist_asin,padding='post',value=0)
    X={'reviewerID':reviewerID,'asin':asin,'hist_asin':hist_asin,'length_name':seq_length}

    feature_columns=[SparseFeat('reviewerID',user_count,embedding_dim=32),SparseFeat('asin',item_count,embedding_dim=32)]
    feature_columns+=[VarLenSparseFeat(SparseFeat('hist_asin',item_count,embedding_dim=32,embedding_name='hist_asin'),maxlen=max_len,length_name="length_name")]
    behaviour_list=['asin']
    if method=='DeepFM':
        behaviour_list=[SparseFeat('asin',item_count,embedding_dim=32)]
    return X,y,feature_columns,behaviour_list


def test_DIN():
    x, y, feature_columns, behavior_feature_list = get_config()# get_xy_fd()
    print('?????????')
    model = DIN(feature_columns, behavior_feature_list)

    model.load_weights(folder+r'/param/my_model')
    model.compile('adam', 'binary_crossentropy',
                  metrics=[metrics.binary_accuracy])

    history = model.fit(x, y, verbose=1, epochs=1, validation_split=0.1,batch_size=64)
    model.save_weights(folder+r'/param/my_model')

def auc(y_true, y_pred):
    '''
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
    '''
    y_true=y_true.to
    return auc
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

'''
def test_DeepFM():
    x, y, feature_columns, behavior_feature_list =get_xy_fd('DeepFM') #get_config('DeepFM')


    model=DeepFM(behavior_feature_list,feature_columns)
    model.load_weights(folder + r'/param/DeepFM')
    model.compile('adam', 'binary_crossentropy',
                  metrics=[metrics.binary_accuracy])

    history = model.fit(x, y, verbose=1, epochs=1, validation_split=0.1, batch_size=64)
    model.save_weights(folder + r'/param/DeepFM')
'''
def test_DeepFM():
    x, y, feature_columns, behavior_feature_list =get_xy_fd('DeepFM') #get_config('DeepFM')


    model=DeepFM(behavior_feature_list,feature_columns)
    #model.load_weights(folder + r'/param/DeepFM')
    model.compile('adam', 'binary_crossentropy',
                  metrics=[metrics.binary_accuracy])

    history = model.fit(x, y, verbose=1, epochs=1, validation_split=0.1, batch_size=64)
    #model.save_weights(folder + r'/param/DeepFM')



def test_dien():
    def get_xy_fd(use_neg=False, hash_flag=False):
        feature_columns = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                           SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                           SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                           SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                           DenseFeat('pay_score', 1)]

        feature_columns += [
            VarLenSparseFeat(
                SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
                             length_name="seq_length")]

        behavior_feature_list = ["item_id", "cate_id"]
        uid = np.array([0, 1, 2])
        ugender = np.array([0, 1, 0])
        iid = np.array([1, 2, 3])  # 0 is mask value
        cate_id = np.array([1, 2, 2])  # 0 is mask value
        score = np.array([0.1, 0.2, 0.3])

        hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        hist_cate_id = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])

        behavior_length = np.array([3, 3, 2])

        feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                        'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                        'pay_score': score, "seq_length": behavior_length}

        if use_neg:
            feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
            feature_dict['neg_hist_cate_id'] = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])
            feature_columns += [
                VarLenSparseFeat(
                    SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                    maxlen=4, length_name="seq_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                                 maxlen=4, length_name="seq_length")]

        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
        y = np.array([1, 0, 1])
        return x, y, feature_columns, behavior_feature_list
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    USE_NEG = True
    x, y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=USE_NEG)

    model = DIEN(feature_columns, behavior_feature_list,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.6, gru_type="AUGRU", use_negsampling=USE_NEG)
    model.summary()
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)