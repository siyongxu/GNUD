import tensorflow as tf
from abc import abstractmethod
import numpy as np

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors):
        print(neighbor_vectors)
        b = tf.reduce_max(neighbor_vectors, -1)
        c = b > 0.0
        d = tf.reduce_sum(tf.cast(c, tf.float32), -1, keepdims=True)
        d = tf.nn.bias_add(d, [1e-10])
        print('agg',b,c,d)
        e = tf.tile(d, [1, 1, self.dim])
        neighbors_aggregated = tf.reduce_sum(neighbor_vectors, axis=2)/e
        # neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)
        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)
        print(self_vectors,neighbors_agg)
        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)
        print('*****',self_vectors, neighbors_agg)
        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])  # [128*31,128]
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)




class RoutingLayer():
    def __init__(self, layers, out_caps, cap_sz, batch_size, drop, inp_caps=None, name=None, tau=1.0):

        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.batch_size = batch_size
        self.tau = tau
        self.drop = drop

        self.cap_sz = cap_sz

        self.d, self.k = out_caps * cap_sz, out_caps
        self._cache_zero_d = tf.zeros([1, self.d])
        self._cache_zero_k = tf.zeros([1, self.k])

        if inp_caps is not None:
            self.inp_caps = inp_caps
            if layers == 1:
                with tf.variable_scope('Linear-1'):
                    stdv = 1. / tf.sqrt(tf.cast(self.d, tf.float32))
                    self.w1 = tf.get_variable(shape=[inp_caps * cap_sz,  cap_sz * out_caps], initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv), name='weights')
                    self.b1 = tf.get_variable(shape=[cap_sz * out_caps], initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv), name='bias')
            if layers == 2:
                with tf.variable_scope('Linear-2'):
                    stdv = 1. / tf.sqrt(tf.cast(self.d, tf.float32))
                    self.w2 = tf.get_variable(shape=[inp_caps * cap_sz, cap_sz * out_caps],initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv),name='weights')
                    self.b2 = tf.get_variable(shape=[cap_sz * out_caps],initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv),name='bias')

    def drop_out(self, x):
        return tf.nn.dropout(x, keep_prob=1-self.drop)

    def rout(self, self_vectors, neighbor_vectors, max_iter):

        if hasattr(self, 'w1'):
            self_z = tf.nn.relu(tf.matmul(tf.reshape(self_vectors, [-1, self.inp_caps*self.cap_sz]), self.w1) + self.b1)
            neighbor_z = tf.nn.relu(tf.matmul(tf.reshape(neighbor_vectors, [-1, self.inp_caps*self.cap_sz]), self.w1) + self.b1)
        elif hasattr(self, 'w2'):
            self_z = tf.nn.relu(
                tf.matmul(tf.reshape(self_vectors, [-1, self.inp_caps * self.cap_sz]), self.w2) + self.b2)
            neighbor_z = tf.nn.relu(tf.matmul(tf.reshape(neighbor_vectors, [-1, self.inp_caps * self.cap_sz]),
                                              self.w2) + self.b2)
        else:  # 第一层不激活
            self_z = tf.reshape(self_vectors, [-1, self.d])
            neighbor_z = tf.reshape(neighbor_vectors, [-1, self.d])
        self_size, neighbor_size = self_vectors.shape, neighbor_vectors.shape
        self_n, neighbor_n = self_size[0] * self_size[1], neighbor_size[0]*neighbor_size[1]
        d, k, delta_d = self.d, self.k, self.d // self.k

        self_z_n = tf.nn.l2_normalize(tf.reshape(tf.reshape(self_z, [self.batch_size, -1, d]),
                                                 [self.batch_size, -1, k, delta_d]), axis=3)
        neighbor_z_n = tf.nn.l2_normalize(tf.reshape(tf.reshape(neighbor_z, [self.batch_size, -1, d]),
                                                 [self.batch_size, -1, k, delta_d]), axis=3)
        #self_z_n = tf.reshape(self_z_n, [self.batch_size, -1, d])
        neighbor_z_n = tf.reshape(neighbor_z_n, [self.batch_size, -1, neighbor_size[-2], k, delta_d])

        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = tf.tile(tf.reshape(self._cache_zero_k, [1, 1, 1, self.k]), [self.batch_size, neighbor_size[-3], neighbor_size[-2], 1])
            else:
                p = tf.reduce_sum(neighbor_z_n * tf.reshape(u, [self.batch_size, -1, 1, k, delta_d]), axis=-1)
            p = tf.nn.softmax(p / self.tau, axis=-1)

            u = tf.reduce_sum(neighbor_z_n * tf.reshape(p, [self.batch_size, -1, neighbor_size[-2], k, 1]), axis=2)
            u += self_z_n
            if clus_iter < max_iter - 1:
                u = tf.nn.l2_normalize(u, axis=-1)


        return self.drop_out(tf.nn.relu(tf.reshape(u, [self.batch_size, -1, d])))

