import tensorflow as tf
import numpy as np

alpha = 0.01
embedding_size = 16
cate_size = 10
batch_size = 32

cate2_batch = [np.random.randint(0, cate_size) for _ in range(batch_size)]

graph = tf.Graph()
with graph.as_default():
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_size, embedding_size])
    cate2_emb = tf.nn.embedding_lookup(cate_emb_w, cate2_batch)

    ...

    cur_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads_vals = opt.compute_gradients(cur_loss, [cate_emb_w])

    for i, (grad, val) in enumerate(grads_vals):
        grad_buffer = [[] for _ in range(cate_size)]
        cnt_buffer = [[] for _ in range(cate_size)]

        # 记录梯度
        for value in cate2_batch:
            g_1 = tf.nn.embedding_lookup(grad.values, value)
            cnt_buffer[value].append(1)
            grad_buffer[value].append(g_1)

        # 对梯度求平均
        for ind in range(len(grad_buffer)):
            if len(grad_buffer[ind]) > 0:
                grad_buffer[ind] = tf.reduce_mean(grad_buffer[ind], axis=0)
            else:
                grad_buffer[ind] = tf.gather(grad.values, ind)

        # 根据公式更新梯度
        new_grad = [0] * cate_size
        for ind in range(cate_size):
            new_grad[ind] = grad_buffer[ind]
            cnt_buffer[ind] = [sum(cnt_buffer[ind])] * embedding_size
        new_grad += tf.divide(val, cnt_buffer) * alpha

        grads_vals[i] = (new_grad, val)
    train_op = opt.apply_gradients(grads_vals)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(train_op))
