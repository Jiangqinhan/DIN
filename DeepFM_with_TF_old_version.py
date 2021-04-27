import tensorflow as tf
import pickle
import time
from build_dataset import DataInput

class DeepFM:
    def __init__(self, user_count, item_count, cate_count, cate_list):
        self.user_id = tf.placeholder(tf.int32, [None, ])  # [B]
        self.item_id = tf.placeholder(tf.int32, [None, ])  # [B]
        self.label = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_item = tf.placeholder(tf.int32, [None, None])  # [B,length]
        self.sequnce_length = tf.placeholder(tf.int64, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])  # 学习率


        embedding_size = 128

        # 除以2的是因为 要把两部分组合起来作为item的embedding
        user_emb_w = tf.get_variable("user_emb_w", [user_count, embedding_size])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, embedding_size // 2])
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, embedding_size // 2])
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int32)

        u_emb = tf.nn.embedding_lookup(user_emb_w, self.user_id)  # [B,Embedding_size]
        item_category = tf.gather(cate_list, self.item_id)
        item_emb = tf.concat(
            [tf.nn.embedding_lookup(item_emb_w, self.item_id), tf.nn.embedding_lookup(cate_emb_w, item_category)],
            axis=1)  # [B,Embedding_size]
        item_bias = tf.gather(item_b, self.item_id)

        hist_category = tf.gather(cate_list, self.hist_item)  # [B,input_length]
        hist_emb = tf.concat(
            [tf.nn.embedding_lookup(item_emb_w, self.hist_item), tf.nn.embedding_lookup(cate_emb_w, hist_category)],
            axis=2)  # [B,input_length,Embeddingsize]

        # 历史行为加权求平均,将其记为u_emb非常离谱
        mask = tf.sequence_mask(self.sequnce_length, tf.shape(hist_emb)[1], dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, embedding_size])
        hist = hist_emb * mask
        hist = tf.reduce_sum(hist, axis=1,keep_dims=False)
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sequnce_length,1), [1, embedding_size]), tf.float32))
        u_emb = hist

        # dnn部分
        dnn = tf.concat([u_emb, item_emb], axis=-1)
        dnn = tf.layers.batch_normalization(inputs=dnn, name='batch_norm_1')
        dnn = tf.layers.dense(dnn, 80, activation=tf.nn.sigmoid, name='f1')
        dnn = tf.layers.dense(dnn, 40, activation=tf.nn.sigmoid, name='f2')
        dnn = tf.layers.dense(dnn, 1, activation=None, name='f3')

        # fm部分
        fm = tf.concat([u_emb * item_emb, tf.gather(u_emb, [0], axis=-1), tf.gather(item_emb, [0], axis=-1)], axis=-1)
        fm = tf.reduce_sum(fm, axis=-1)
        dnn=tf.reshape(dnn,[-1])
        fm=tf.reshape(fm,[-1])
        item_bias=tf.reshape(item_bias,[-1])
        self.logits = item_bias + fm + dnn
        # 设置优化部分
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        trainable_params = tf.trainable_variables()
        gradient = tf.gradients(self.loss, trainable_params)
        # 防止梯度过大
        clip_gradients,_ = tf.clip_by_global_norm(gradient, 5)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, batch, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.user_id: batch[0],
            self.item_id: batch[1],
            self.label: batch[2],
            self.hist_item: batch[3],
            self.sequnce_length: batch[4],
            self.lr: lr
        })
        return loss


if __name__ == "__main__":
    folder = r"D:\Amozon_data_set"
    batch_size=32
    with open(folder + r"/dataset.pkl", 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)


    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model=DeepFM(user_count, item_count, cate_count, cate_list)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        lr = 0.001
        start_time = time.time()
        for _,batch in DataInput(train_set,batch_size):
            loss=model.train(sess,batch,lr)
            print(loss)


