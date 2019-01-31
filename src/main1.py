import numpy as np
import tensorflow as tf


# input Graph
def inp_graph():
    g = np.array([[0.0, 0.2, 0, 0, 0, 0, 0.3, 0.5],
                  [0.4, 0, 0.3, 0.1, 0.1, 0, 0.1, 0.0],
                  [0.2, 0, 0, 0, 0, 0, 0.3, 0.5],
                  [0.0, 0, 0.1, 0.1, 0.25, 0, 0.3, 0.25],
                  [0.1, 0, 0.1, 0, 0.2, 0, 0.1, 0.5],
                  [0.1, 0, 0.1, 0, 0.2, 0, 0.1, 0.5],
                  [0.2, 0, 0, 0, 0, 0, 0.3, 0.5],
                  [0.2, 0, 0, 0, 0, 0, 0.3, 0.5],
                  ])
    return g


class graph():
    def __init__(self, g, p, q):
        self.adj = g
        self.p = p
        self.q = q
        self.q_inv = 1 / q
        self.p_inv = 1 / p
        # self.get_transformed_matrix(self)

    def nbrs(self, i):
        i_row = self.adj[i]
        return np.nonzero(i_row)

    # For 2nd order RW
    # [i][j]
    #   -> if i reached from j : 1/p
    #   -> if i from nbr(j) : 1
    #   -> else 1/q
    # wl : walk length
    # r : number of walks
    def get_random_walks(self, wl, r):
        walks = []
        for _r in range(r):
            # for each node
            for st in range(self.adj.shape[0]):
                walk = [st]
                # do walk
                cur = st
                prev = None
                prev_nbrs = self.nbrs(st)
                for _w in range(wl - 1):

                    # move to next node
                    if prev == None:
                        # normalize prob
                        probs = self.adj[cur]
                        probs = probs / np.sum(probs)
                        sel = np.random.multinomial(1, probs)
                        idx = np.nonzero(sel)[0][0]
                        prev = cur
                        cur = idx
                    else:
                        probs = self.adj[cur]
                        probs[prev] = probs[prev] / self.p
                        for n in prev_nbrs:
                            probs[n] = probs[n] / self.q
                        probs = probs / np.sum(probs)
                        idx = np.nonzero(np.random.multinomial(1, probs))[0][0]
                        prev = cur
                        cur = idx
                    walk.append(cur)
                    prev_nbrs = self.nbrs(prev)
                walks.append(walk)
        return walks


g = graph(inp_graph(), 3, 0.5)
l = g.get_random_walks(5, 10)
print(l)


# --------------------#

class node2vec:

    def __init__(self, graph_matrix, op_dim):
        self.g_mat = graph_matrix
        self.inp_dim = self.g.hape[0]
        self.op_dim = op_dim
        self.set_model_params()
        self.build_model()
        self.g = graph(self.g_mat, self.p, self.q)
        self.num_nodes = self.g_mat.shape[0]
        return

    def set_model_params(self):
        self.wl = 10
        self.r = 1000
        self.p = 5
        self.q = 2
        self.neg_samples = 10
        self.batch_size = 32
        self.num_epochs = 5
        self.context_size = 2  # select this on either side
        return

    def build_model(self):
        with tf.variable_scope('model'):
            self.x_pos = tf.placeholder(tf.float32, [None, self.context_size * 2, self.inp_dim], name='x')
            self.y_pos = tf.placeholder(tf.float32, [None, self.inp_dim], name='x')
            self.x_neg = tf.placeholder(tf.float32, [None, self.neg_samples, self.inp_dim], name='x')
            self.W = tf.truncated_normal(
                [self.inp_dim, self.op_dim],
                stddev=1
            )

            self.B = tf.truncated_normal(
                [1, self.op_dim],
                stddev=1
            )

            self.emb1 = tf.nn.xw_plus_b(self.y_pos, self.W, self.B)

            tmp = tf.einsum('ijk,kl->ijl', self.x_pos, self.W)
            self.emb2 = tf.nn.bias_add(tmp, self.B)

            tmp = tf.einsum('ijk,kl->ijl', self.x_neg, self.W)
            self.emb3 = tf.nn.bias_add(tmp, self.B)

            tmp = tf.stack([self.y_pos] * self.x_pos.shape[1], axis=1)
            loss1 = tf.losses.cosine_distance(labels=self.y_pos, predictions=tmp, axis=-1)

            tmp = tf.stack([self.y_pos] * self.neg_samples, axis=1)
            loss2 = tf.losses.cosine_distance(labels=self.y_pos, predictions=tmp, axis=-1)
            self.loss = tf.log(loss1) - tf.log(loss2)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
            self.train = self.optimizer.minimize(self.loss)

        return

    # create positive and negative samples
    def get_data(self):
        x_pos = []  # set of surrounding(context nodes)
        y_pos = []  # target node
        x_neg = []  # Random nodes

        walks = self.g.get_random_walks(self, self.wl, self.r)
        # parse rows

        for walk in walks:
            for _idx in range(self.context_size, len(walk) - self.context_size):
                cur = walk[_idx]
                y_pos.append(cur)
                tmp = walk[_idx - self.context_size: _idx]
                x_pos.append(tmp)
                tmp.extend(walk[_idx + 1: _idx + self.context_size])
                mult_p_idx = np.ones(self.num_nodes)
                np.put(mult_p_idx, list(tmp).extend(cur), 0)
                mult_p_idx = mult_p_idx/np.sum(mult_p_idx)
                neg = np.random.multinomial(self.neg_samples, mult_p_idx)
                x_neg.append(neg)
        return

    def train(self):

        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        data = self.get_data()
        bs = self.batch_size
        num_batches = data.shape[0] // bs

        losses = []

        for epoch in range(self.num_epochs):
            _loss = []
            for i in range(num_batches):
                _data = data[i * bs: (i + 1) * bs]

        return
