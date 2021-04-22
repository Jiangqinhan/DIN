from feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
import numpy as np
from din import DIN
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import metrics

def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=10), SparseFeat(
        'gender', 2, embedding_dim=4), SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4), DenseFeat('pay_score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
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
    return x, y, feature_columns, behavior_feature_list

def get_config():
    folder = r"D:\Amozon_data_set"
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
    return X,y,feature_columns,behaviour_list


def test_DIN():
    x, y, feature_columns, behavior_feature_list = get_config() #get_xy_fd()
    print('?????????')

    model = DIN(feature_columns, behavior_feature_list)
    model.compile('adam', 'binary_crossentropy',
                  metrics=[metrics.binary_accuracy])
    history = model.fit(x, y, verbose=1, epochs=5, validation_split=0.5,batch_size=64)




