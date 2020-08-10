import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, RoutingLayer

class Model(object):
    def __init__(self, args, news_title, news_entity, news_group, n_user, n_news):

        n_word = 279215
        self.params = []
        self.use_group = args.use_group
        self.n_filters = args.n_filters
        self.filter_sizes = args.filter_sizes
        self.max_session_len = args.session_len
        self.user_dim = args.user_dim
        self.lr = args.lr
        self.title_len = args.title_len
        self.batch_size = args.batch_size
        self.news_neighbor = args.news_neighbor
        self.user_neighbor = args.user_neighbor
        self.entity_neighbor = args.entity_neighbor
        self.n_iter = args.n_iter
        self.l2_weight = args.l2_weight
        self.cnn_out_size = args.cnn_out_size

        self.news_entity = news_entity
        self.news_group = news_group

        self.title = news_title
        self.ncaps = args.ncaps
        self.dcaps = args.dcaps
        self.nhidden = args.nhidden
        self.dim = self.ncaps * self.nhidden
        self.routit = args.routit
        self.balance = args.balance

        self.n_user = n_user
        self.n_news = n_news

        self.group_embedding = tf.get_variable(name="group_embed", shape=[12, 50], dtype=tf.float32,
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.params.append(self.group_embedding)
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user + 1, self.user_dim], initializer=tf.contrib.layers.xavier_initializer(), name='user_emb_matrix')
        self.word_emb_matrix = tf.get_variable(
            shape=[n_word + 1, 50], initializer=tf.truncated_normal_initializer(stddev=0.1), name='word_emb_matrix')
        self.params.append(self.user_emb_matrix)
        self.params.append(self.word_emb_matrix)

        self.filter_shape_item = [40, 20, 1, 8]
        self.input_size_item = 10 * 8 * 8
        self.filter_shape_title = [2, 20, 1, 8]
        self.input_size_title = 4 * 8 * 8
        self.filter_shape = [2, 8, 1, 4]
        self.cat_size = 7 * 30 * 4

        self.build_inputs()  # placeholder

        self.Routing = RoutingLayer

        self.build_model()
        self.build_train()

    def build_inputs(self):
        self.dropout_rate = tf.placeholder(tf.float32)

        self.user_indices = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='user_indices')
        self.news_indices = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='news_indices')  # 一个batch大小
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='labels')

        self.user_news = tf.placeholder(dtype=tf.int32, shape=[self.n_user, self.news_neighbor], name='user_news')
        self.news_user = tf.placeholder(dtype=tf.int32, shape=[self.n_news, self.user_neighbor], name='user_news')

    def build_model(self):

        self.user_emb_matrix = tf.nn.l2_normalize(self.user_emb_matrix, axis=-1)
        self.word_emb_matrix = tf.nn.l2_normalize(self.word_emb_matrix, axis=-1)

        newsvec, uservec = self.get_neighbors(self.news_indices, self.user_indices)

        self.news_embeddings, self.user_embeddings, self.aggregators = self.aggregate(newsvec, uservec)
        self.scores = tf.squeeze(self.simple_dot_net(self.user_embeddings, self.news_embeddings))
        self.scores_normalized = tf.sigmoid(self.scores)
        self.predict_label = tf.cast(self.scores > 0.5, tf.int32)
        print('build tensor graph over!')
    def build_train(self):
        total_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores)

        self.base_loss = tf.reduce_mean(total_loss)

        self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
        for param in self.params:  #
            self.l2_loss = tf.add(self.l2_loss, tf.nn.l2_loss(param))
        for i, aggregator in enumerate(self.aggregators):
            if i == 0:
                continue
            if i == 1:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.w1)
            if i == 2:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.w2)
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.user_weights) + tf.nn.l2_loss(self.item_weights)
        infer_loss, ret_w = self.infer_loss(self.user_embeddings, self.news_embeddings)
        self.loss = (1-self.balance) * self.base_loss + self.balance * infer_loss + self.l2_weight * self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def simple_dot_net(self, x, y):
        caps = self.ncaps - (self.n_iter - 1) * self.dcaps
        with tf.variable_scope("last_map"):
            last_w = tf.get_variable(shape=[caps * self.nhidden, caps * self.nhidden],
                                     initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            last_b = tf.get_variable(shape=[caps * self.nhidden], initializer=tf.zeros_initializer(), name='bias')

            x_map = tf.matmul(tf.reshape(x[-1], [self.batch_size, -1]), last_w) + last_b
            y_map = tf.matmul(tf.reshape(y[-1], [self.batch_size, -1]), last_w) + last_b
            print(x_map.shape)
        output = tf.reduce_sum(x_map*y_map, axis=-1)
        print(output.shape)  # [batch,]
        return output

    def infer_loss(self, x, y):
        caps = self.ncaps - (self.n_iter - 1) * self.dcaps
        with tf.variable_scope("ret_-2"):
            ret_uw = tf.get_variable(shape=[self.nhidden, caps],
                                         initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            ret_ub = tf.get_variable(shape=[caps], initializer=tf.zeros_initializer(), name='bias')
        x_class = tf.matmul(tf.reshape(x[-1], [-1, self.nhidden]), ret_uw) + ret_ub
        y_class = tf.matmul(tf.reshape(y[-1], [-1, self.nhidden]), ret_uw) + ret_ub
        label = tf.tile(tf.eye(caps), [self.batch_size, 1])
        user_infer_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=x_class)))
        news_infer_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_class)))

        loss = user_infer_loss + news_infer_loss

        return loss, ret_uw
    def get_neighbors(self, news_seeds, user_seeds):
        news_seeds = tf.expand_dims(news_seeds, axis=1)
        user_seeds = tf.expand_dims(user_seeds, axis=1)
        news = [news_seeds]
        user = [user_seeds]
        news_vectors = []
        user_vectors = []
        n = self.news_neighbor
        u = self.user_neighbor

        with tf.variable_scope("user_Map"):
            stdv = 1. / tf.sqrt(tf.cast(self.dim, tf.float32))
            self.user_weights = tf.get_variable(shape=[self.user_dim, self.dim], initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv), name='weights')
            self.user_bias = tf.get_variable(shape=[self.dim], initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv), name='bias')
        with tf.variable_scope("item_Map"):
            stdv = 1. / tf.sqrt(tf.cast(self.dim, tf.float32))
            self.item_weights = tf.get_variable(shape=[self.cnn_out_size, self.dim], initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv), name='weights')
            self.item_bias = tf.get_variable(shape=[self.dim], initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv), name='bias')

        news_hop_vectors = tf.reshape(self.convolution(news[0]), [-1, self.cnn_out_size])
        news_hop_vectors = tf.matmul(news_hop_vectors, self.item_weights) + self.item_bias

        news_vectors.append(tf.reshape(news_hop_vectors, [self.batch_size,-1, self.dim]))
        news_neighbors = tf.nn.embedding_lookup(self.news_user, news[0][:, 0])
        news.append(news_neighbors)
        print("news---hop----0", news, news_vectors)

        user_hop_vectors = tf.reshape(tf.nn.embedding_lookup(self.user_emb_matrix, user[0]), [-1, self.user_dim])
        user_hop_vectors = tf.matmul(user_hop_vectors,self.user_weights) + self.user_bias
        user_vectors.append(tf.reshape(user_hop_vectors,[self.batch_size, -1, self.dim]))
        user_neighbors = tf.nn.embedding_lookup(self.user_news, user[0][:, 0])
        user.append(user_neighbors)
        print("user---hop----0", user, user_vectors)

        if self.n_iter >= 1:
            news_hop_vectors = tf.reshape(tf.nn.embedding_lookup(self.user_emb_matrix, news[1][:, :u]),[-1, self.user_dim])
            news_hop_vectors = tf.matmul(news_hop_vectors, self.user_weights) + self.user_bias
            news_hop_vectors = tf.reshape(news_hop_vectors, [self.batch_size, -1, self.dim])
            news_neighbors = tf.reshape(tf.gather(self.user_news, news[1][:, :u]), [self.batch_size, -1])
            news_vectors.append(news_hop_vectors)
            news.append(news_neighbors)
            print("news---hop----1", news, news_vectors)

            user_hop_vectors = tf.reshape(self.convolution(user[1]), [-1, self.cnn_out_size])
            user_hop_vectors = tf.matmul(user_hop_vectors, self.item_weights) + self.item_bias
            user_hop_vectors = tf.reshape(user_hop_vectors, [self.batch_size, -1, self.dim])
            user_neighbors = tf.reshape(tf.gather(self.news_user, user[1][:, :n]), [self.batch_size, -1])
            user_vectors.append(user_hop_vectors)
            user.append(user_neighbors)  #
            print("news---hop----1", user, user_vectors)

        if self.n_iter >= 2:
            news_hop_vectors = tf.reshape(self.convolution(news[2]), [-1, self.cnn_out_size])
            news_hop_vectors = tf.matmul(news_hop_vectors, self.item_weights) + self.item_bias
            news_hop_vectors = tf.reshape(news_hop_vectors, [self.batch_size, -1, self.dim])
            news_neighbors = tf.gather(self.news_user, news[2])
            news_neighbors = tf.reshape(news_neighbors, [self.batch_size, -1])
            news_vectors.append(news_hop_vectors)
            news.append(news_neighbors)
            print("news---hop----2", news, news_vectors)


            user_hop_vectors = tf.reshape(tf.nn.embedding_lookup(self.user_emb_matrix, user[2]),
                       [-1, self.user_dim])
            user_hop_vectors = tf.matmul(user_hop_vectors, self.user_weights) + self.user_bias
            user_hop_vectors = tf.reshape(user_hop_vectors, [self.batch_size, -1, self.dim])
            user_neighbors = tf.reshape(tf.gather(self.user_news, user[2]), [self.batch_size, -1])
            user_vectors.append(user_hop_vectors)
            user.append(user_neighbors)
            print("user---hop----2", user, user_vectors)


        if self.n_iter >= 3:
            j = 0
            while j < news[3].shape[1]:
                if j == 0:
                    news_hop_vectors = tf.reshape(tf.nn.embedding_lookup(self.user_emb_matrix, news[3][:, :u])
                               , [-1, self.user_dim])
                    news_hop_vectors = tf.matmul(news_hop_vectors, self.user_weights) + self.user_bias
                    news_hop_vectors = tf.reshape(news_hop_vectors, [self.batch_size, -1, self.dim])
                    j += u
                else:
                    t = tf.reshape(tf.nn.embedding_lookup(self.user_emb_matrix, news[3][:, j:j + u]),
                               [-1, self.user_dim])
                    t = tf.matmul(t, self.user_weights) + self.user_bias
                    t = tf.reshape(t, [self.batch_size, -1, self.dim])
                    news_hop_vectors = tf.concat([news_hop_vectors, t], axis=1)
                    j += u
            news_vectors.append(news_hop_vectors)

            print("news---hop----3", news, news_vectors)

            i = 0
            while i < user[3].shape[1]:
                if i == 0:
                    user_hop_vectors = tf.reshape(self.convolution(user[3][:, :n]), [-1, self.cnn_out_size])
                    user_hop_vectors = tf.matmul(user_hop_vectors, self.item_weights) + self.item_bias
                    user_hop_vectors = tf.reshape(user_hop_vectors, [self.batch_size, -1, self.dim])
                    i += n
                else:
                    t = tf.reshape(self.convolution(user[3][:, i:i + n]), [-1, self.cnn_out_size])
                    t = tf.matmul(t, self.user_weights) + self.user_bias
                    t = tf.reshape(t, [self.batch_size, -1, self.dim])
                    user_hop_vectors = tf.concat([user_hop_vectors, t], axis=1)
                    i += n
            user_vectors.append(user_hop_vectors)
            #user.append(user_neighbors)
            print("user---hop---3", user, user_vectors)
        return news_vectors, user_vectors

    # feature propagation
    def aggregate(self, news_vectors, user_vectors):

        conv_ls = []  # store all routing_layer
        conv = None
        inp_caps, out_caps = None, self.ncaps
        cur_dim = self.dim

        news = []
        user = []
        for i in range(self.n_iter):
            print("layer--", i)
            conv = self.Routing(i, out_caps, self.nhidden, self.batch_size, self.dropout_rate,
                                inp_caps)


            conv_ls.append(conv)

            news_vectors_next_iter = []
            user_vectors_next_iter = []


            for hop in range(self.n_iter - i):
                # shape = [self.batch_size, -1, n_neighbor, self.dim]

                if hop % 2 == 0:
                    if inp_caps == None:
                        news_shape = [self.batch_size, -1, self.user_neighbor, self.dim]
                        user_shape = [self.batch_size, -1, self.news_neighbor, self.dim]
                    else:
                        news_shape = [self.batch_size, -1, self.user_neighbor, inp_caps * self.nhidden]
                        user_shape = [self.batch_size, -1, self.news_neighbor, inp_caps * self.nhidden]
                else:
                    if inp_caps == None:
                        news_shape = [self.batch_size, -1, self.news_neighbor, self.dim]
                        user_shape = [self.batch_size, -1, self.user_neighbor, self.dim]
                    else:
                        news_shape = [self.batch_size, -1, self.news_neighbor, inp_caps * self.nhidden]
                        user_shape = [self.batch_size, -1, self.user_neighbor, inp_caps * self.nhidden]
                print("news--hop", hop, news_vectors[hop], tf.reshape(news_vectors[hop+1], news_shape))
                print("user--hop", hop, user_vectors[hop], tf.reshape(user_vectors[hop + 1], user_shape))

                news_vectors[hop] = news_vectors[hop]
                news_vectors[hop+1] = news_vectors[hop+1]
                news_vector = conv.rout(self_vectors=news_vectors[hop],
                                neighbor_vectors=tf.reshape(news_vectors[hop+1], news_shape), max_iter=self.routit)
                user_vectors[hop] = user_vectors[hop]
                user_vectors[hop+1] = user_vectors[hop+1]
                user_vector = conv.rout(self_vectors=user_vectors[hop],
                                neighbor_vectors=tf.reshape(user_vectors[hop + 1], user_shape), max_iter=self.routit)


                news_vectors_next_iter.append(news_vector)
                user_vectors_next_iter.append(user_vector)

            news_vectors = news_vectors_next_iter
            user_vectors = user_vectors_next_iter

            news.append(tf.reshape(tf.reshape(news_vectors[0], [self.batch_size, -1]), [self.batch_size, out_caps, self.nhidden]))
            user.append(tf.reshape(tf.reshape(user_vectors[0], [self.batch_size, -1]), [self.batch_size, out_caps, self.nhidden]))

            cur_dim += out_caps * self.nhidden
            inp_caps, out_caps = out_caps, max(1, out_caps - self.dcaps)

        return news, user, conv_ls

    def convolution(self, inputs):
        title_lookup = tf.reshape(tf.nn.embedding_lookup(self.title, inputs), [-1, self.title_len])
        title_embed = tf.expand_dims(tf.nn.embedding_lookup(self.word_emb_matrix, title_lookup), -1)  #

        item_lookup = tf.reshape(tf.nn.embedding_lookup(self.news_entity, inputs), [-1, 40])
        group_lookup = tf.reshape(tf.nn.embedding_lookup(self.news_group, inputs), [-1, 40])
        item_embed = tf.expand_dims(tf.nn.embedding_lookup(self.word_emb_matrix, item_lookup), 2)  #
        group_embed = tf.expand_dims(tf.nn.embedding_lookup(self.group_embedding, group_lookup), 2)  #
        item_group_embed = tf.expand_dims(
            tf.reshape(tf.concat((item_embed, group_embed), 2), [-1, 80, 50]), -1)

        with tf.variable_scope("conv-maxpool-item-group", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            W_item = tf.get_variable(name='W', shape=self.filter_shape_item, dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_item = tf.get_variable(name='b', shape=[8], dtype=tf.float32)
            if W_item not in self.params:
                self.params.append(W_item)
            if b_item not in self.params:
                self.params.append(b_item)
            conv_item = tf.nn.conv2d(
                item_group_embed,
                W_item,
                strides=[1, 2, 2, 1],
                padding="VALID",
                name="conv")
            h_item = tf.nn.relu(tf.nn.bias_add(conv_item, b_item), name="relu")
            pooled_item = tf.nn.max_pool(
                h_item,
                ksize=[1, 3, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name="pool")
            self.pool_item = tf.reshape(pooled_item, [self.batch_size, -1, self.input_size_item])

        with tf.variable_scope("conv-maxpool-title", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            W_title = tf.get_variable(name='W', shape=self.filter_shape_title, dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_title = tf.get_variable(name='b', shape=[8], dtype=tf.float32)
            if W_title not in self.params:
                self.params.append(W_title)
            if b_title not in self.params:
                self.params.append(b_title)
            conv_title = tf.nn.conv2d(
                title_embed,
                W_title,
                strides=[1, 2, 2, 1],
                padding="VALID",
                name="conv")
            h_title = tf.nn.relu(tf.nn.bias_add(conv_title, b_title), name="relu")
            pooled_title = tf.nn.max_pool(
                h_title,
                ksize=[1, 2, 1, 1],
                strides=[1, 1, 2, 1],
                padding='VALID',
                name="pool")
            pool_title = tf.reshape(pooled_title, [self.batch_size, -1, self.input_size_title])

        pooled = tf.concat((self.pool_item, pool_title), -1)
        pool = tf.layers.dense(pooled, self.cnn_out_size, activation=tf.nn.relu)

        return pool

    def train(self, sess, feed_dict):
        o, l, n, u, labels, scores= sess.run([self.optimizer, self.loss, self.news_embeddings, self.user_embeddings, self.labels, self.scores_normalized], feed_dict)

        predict = [1 if i >= 0.5 else 0 for i in scores]
        auc = roc_auc_score(y_true=labels, y_score=scores)
        f1 = f1_score(labels, predict)
        return o, l, n, u, auc, f1

    def eval(self, sess, feed_dict):

        labels, scores, los = sess.run([self.labels, self.scores_normalized, self.loss], feed_dict)

        predict = [1 if i >= 0.5 else 0 for i in scores]
        auc = roc_auc_score(y_true=labels, y_score=scores)
        f1 = f1_score(labels, predict)

        return auc, f1,  predict
