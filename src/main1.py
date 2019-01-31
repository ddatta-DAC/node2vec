import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx


# input Graph
# Simple example : Zachary's Karate Club
def inp_graph():
    G = nx.karate_club_graph()
    num_nodes = len(G._adj)
    g = np.diag([1.0] * num_nodes)

    for k, v in G._adj.items():
        idx = k
        for n in v.keys():
            g[idx][n] = 1.0
    print(g)
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

                        if np.sum(probs) != 0:
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

                        if(np.sum(probs)>0) :
                            probs = probs / np.sum(probs)
                        idx = np.nonzero(np.random.multinomial(1, probs))[0][0]
                        prev = cur
                        cur = idx
                    walk.append(cur)
                    prev_nbrs = self.nbrs(prev)[0]
                walks.append(walk)
        return walks


# g = graph(inp_graph(), 3, 0.5)
# l = g.get_random_walks(7, 10)
# print(l)


# --------------------#

class node2vec:

    def __init__(self, graph_matrix, op_dim):
        self.g_mat = graph_matrix
        self.inp_dim = self.g_mat.shape[0]
        self.op_dim = op_dim
        self.set_model_params()
        self.g = graph(self.g_mat, self.p, self.q)
        self.num_nodes = self.g_mat.shape[0]
        self.build_model()
        return

    def set_model_params(self):
        self.wl = 10
        self.r = 1000
        self.p = 5
        self.q = 2
        self.neg_samples = 10
        self.batch_size = 32
        self.num_epochs = 10
        self.context_size = 2  # select this on either side
        self.show_loss = True
        return

    def build_model(self):
        with tf.variable_scope('model'):
            self.x_pos_inp = tf.placeholder(tf.int32, [None, self.context_size * 2], name='x_pos')
            self.y_pos_inp = tf.placeholder(tf.int32, [None, 1], name='y_pos')
            self.x_neg_inp = tf.placeholder(tf.int32, [None, self.neg_samples], name='x_neg')

            # do one hot decoding
            self.x_pos = tf.one_hot(
                indices=self.x_pos_inp,
                depth=self.inp_dim
            )

            self.y_pos = tf.one_hot(
                indices=self.y_pos_inp,
                depth=self.inp_dim
            )

            self.x_neg = tf.one_hot(
                indices=self.x_neg_inp,
                depth=self.inp_dim
            )
            # declare weights #
            print('Shape : x_pos  x_neg y_pos ', self.x_pos.shape, self.x_neg.shape, self.y_pos.shape)

            initial = tf.truncated_normal([self.inp_dim, self.op_dim], stddev=0.1)
            self.W = tf.Variable(initial)
            initial = tf.truncated_normal([1, self.op_dim], stddev=0.1)
            self.B = tf.Variable(initial)

            # self.emb1 = tf.nn.xw_plus_b(self.y_pos, self.W, self.B)

            self.emb1 = tf.einsum('ijk,kl->ijl', self.y_pos, self.W)
            self.emb1 = tf.add(self.emb1, self.B)

            tmp = tf.einsum('ijk,kl->ijl', self.x_pos, self.W)
            self.emb2 = tf.add(tmp, self.B)

            tmp = tf.einsum('ijk,kl->ijl', self.x_neg, self.W)
            self.emb3 = tf.add(tmp, self.B)

            print('Shape : emb1  emb2 emb3 ', self.emb1.shape, self.emb2.shape, self.emb3.shape)

            # ------- Loss function --------- #
            # Expand tensor to do dot product
            tmp1 = tf.stack([self.emb1] * self.x_pos.shape[1], axis=1)
            tmp1 = tf.squeeze(tmp1, axis=2)
            t1 = tf.nn.l2_normalize(self.emb2, -1)
            t2 = tf.nn.l2_normalize(tmp1, -1)
            cs1 = tf.reduce_sum(tf.multiply(t1, t2))

            tmp2 = tf.stack([self.emb1] * self.neg_samples, axis=1)
            tmp2 = tf.squeeze(tmp2, axis=2)
            # do dot product
            t1 = tf.nn.l2_normalize(self.emb3, -1)
            t2 = tf.nn.l2_normalize(tmp2, -1)
            cs2 = tf.multiply(t1, t2)
            cs2 = tf.reduce_sum(cs2,axis=-1)
            # do exp
            cs2 = tf.math.exp(cs2)
            cs2 = -tf.log(tf.reduce_sum(cs2))
            loss = -(cs1 - cs2)

            self.loss = loss
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
            self.train = self.optimizer.minimize(self.loss)

    # create positive and negative samples
    def get_data(self):
        x_pos = []  # set of surrounding(context nodes)
        y_pos = []  # target node
        x_neg = []  # Random nodes

        walks = self.g.get_random_walks(self.wl, self.r)
        # parse rows

        for walk in walks:
            for _idx in range(self.context_size, len(walk) - self.context_size):
                cur = walk[_idx]
                y_pos.append([cur])
                tmp = walk[_idx - self.context_size: _idx]
                tmp.extend(walk[_idx + 1: _idx + self.context_size + 1])
                x_pos.append(tmp)

                mult_p_idx = np.ones(self.num_nodes)
                exclude = list(tmp)
                exclude.append(cur)

                np.put(mult_p_idx, exclude, 0)
                mult_p_idx = mult_p_idx / np.sum(mult_p_idx)
                neg = [np.nonzero(np.random.multinomial(1, mult_p_idx))[0][0] for _ in range(self.neg_samples)]
                x_neg.append(neg)

        x_pos = np.array(x_pos)
        x_neg = np.array(x_neg)
        y_pos = np.array(y_pos)
        print(x_pos.shape)
        print(y_pos.shape)
        print(x_neg.shape)
        return x_pos, y_pos, x_neg

    def train_model(self):

        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        x_pos, y_pos, x_neg = self.get_data()
        bs = self.batch_size
        num_batches = x_pos.shape[0] // bs

        losses = []

        for epoch in range(self.num_epochs):
            _loss = []
            for i in range(num_batches):
                data_x_pos = x_pos[i * bs: (i + 1) * bs]
                data_x_neg = x_neg[i * bs: (i + 1) * bs]
                data_y_pos = y_pos[i * bs: (i + 1) * bs]

                loss, _ = self.sess.run(
                    [self.loss, self.train],
                    feed_dict={
                        self.x_pos_inp: data_x_pos,
                        self.x_neg_inp: data_x_neg,
                        self.y_pos_inp: data_y_pos,
                    })
                _loss.append(loss)
            _loss = np.mean(_loss)
            if epoch % 5 == 0:
                print(_loss)
            losses.append(_loss)

        if self.show_loss == True:
            plt.plot(range(len(losses)), losses, 'r-')
            plt.show()
        return

    def get_emb_dict(self):
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        x = list(range(self.g_mat.shape[0]))
        x = np.reshape(x,[-1,1])
        emb = self.sess.run(
            self.emb1,
            feed_dict={
                self.x_pos_inp: x
            })
        res = {i[0]:i[1] for i in enumerate(emb,0)}
        return res


g_matrix = inp_graph()
n2v = node2vec(g_matrix, 8)
n2v.train_model()
emb_dict = n2v.get_emb_dict()


print (emb_dict)



