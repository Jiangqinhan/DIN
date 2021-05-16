from utils import to_df,build_map
import random
import numpy as np
import pickle


if __name__ == "__main__":
    folder = r"D:\Amozon_data_set"
    '''
    reviews_Electronics=folder+r"\Electronics_5.json"
    meta_Electronics=folder+r"\meta_Electronics.json"

    reviews_df=to_df(reviews_Electronics)
    with open(folder+r"\reviews.pkl","wb") as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    meta_df=to_df(meta_Electronics)
    meta_df=meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df=meta_df.reset_index(drop=True)
    with open(folder+r'\meta.pkl','wb') as f:
        pickle.dump(meta_df,f,pickle.HIGHEST_PROTOCOL)

    with open(folder+r"\reviews.pkl","rb") as f:
        reviews_df=pickle.load(f)
        #选取三列 分别为用户id 商品id 和时间戳
        reviews_df=reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    with open(folder+r'\meta.pkl','rb') as f:
        meta_df=pickle.load(f)
        meta_df=meta_df[['asin','categories']]
        #类别只保留最后一个
        meta_df['categories']=meta_df['categories'].map(lambda x:x[-1][-1])

    asin_map,asin_key=build_map(meta_df,'asin')
    cate_map,cate_key=build_map(meta_df,'categories')
    revi_map,revi_key=build_map(reviews_df,'reviewerID')
    user_count, item_count, cate_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))
    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    #cate_list 物品的标签列表
    cate_list = np.array(meta_df['categories'], dtype='int32')

    with open(folder+r"/remap.pkl",'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
        pickle.dump((user_count, item_count, cate_count, example_count),
                    f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
    '''

    with open(folder+r'/remap.pkl','rb') as f:
        reviews_df=pickle.load(f)
        cate_list=pickle.load(f)
        user_count,item_count,cate_count,example_count=pickle.load(f)


    train_set=[]
    test_set=[]
    print(reviews_df.columns)
    for reviewerID,hist in reviews_df.groupby('reviewerID'):
        pos_list=hist['asin'].tolist()
        def gen_neg():
            neg=pos_list[0]
            while neg in pos_list:
                neg=random.randint(0,item_count-1)
            return neg

        neg_list=[gen_neg()for i in range(len(pos_list))]
        asin_hist=[]
        cate_hist=[]
        for i in range(1,len(pos_list)):
            asin_hist.append(pos_list[i-1])
            cate_hist.append(cate_list[pos_list[i-1]])
            asin_hist_i=asin_hist.copy()
            cate_hist_i=cate_hist.copy()
            if i!=len(pos_list)-1:
                train_set.append((reviewerID,[asin_hist_i,cate_hist_i],[pos_list[i],cate_list[pos_list[i]]],1))
                train_set.append((reviewerID,[asin_hist_i,cate_hist_i],[neg_list[i],cate_list[neg_list[i]]],0))
            else:
                test_set.append((reviewerID,[asin_hist_i,cate_hist_i],[pos_list[i],cate_list[pos_list[i]]],1))
                test_set.append((reviewerID,[asin_hist_i,cate_hist_i],[neg_list[i],cate_list[neg_list[i]]],0))

    random.shuffle(train_set)
    random.shuffle(test_set)
    with open(folder+r"/dataset.pkl",'wb') as f:
        pickle.dump(train_set,f,pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)

    '''
    TODO:
    物品列表里只考虑出现过的样本是否合理？ 在排序的场景下？？？？
    2021/5/15
    '''


