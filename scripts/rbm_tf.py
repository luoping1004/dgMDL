# from __future__ import print_function

import tensorflow as tf
import numpy as np
from util import tf_xavier_init, sample_bernoulli, sample_gaussian, batch_generator

class BBRBM:
    def __init__(self,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.5,
                 weight_cost=0.0002,
                 batch_size=4,
                 n_epoches=20,
                 cd_iter=1):

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_cost = weight_cost
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.cd_iter = cd_iter

    def fit(self, X, X_vali = None):
        self.n_visible = X.shape[1]
        self._build_model()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        for e in range(self.n_epoches):
            if e > 5:
                self.momentum = 0.9
            data = np.array(X)
            for batch in batch_generator(self.batch_size, data):
                self.partial_fit(batch)
        # print('error:', self.get_err(X))
            if e % 5 == 0:
                if X_vali is not None:
                    # print(X[:500,:].shape,X_vali[:500,:].shape)
                    print('gap of epoch',e,'is:',self.free_energy_gap(X[:500,:],X_vali[:500,:]))

        return self

    def _initialize_weights(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=1.0), dtype=tf.float32)
        self.vbias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hbias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_vbias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hbias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

    def sample_v_given_h(self, h_0_p):
        return tf.nn.sigmoid(tf.matmul(sample_bernoulli(h_0_p), tf.transpose(self.w)) + self.vbias)

    def compute_visible_func(self, hidden):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.w)) + self.vbias)

    def free_energy_func(self):
        return -tf.reduce_mean(tf.matmul(self.x, tf.expand_dims(self.vbias, 1))) - tf.reduce_mean(tf.log(1 + tf.exp(tf.matmul(self.x, self.w) + self.hbias)))

    def _build_model(self):
        self._initialize_weights()
        h_0_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hbias)
        v_t = self.sample_v_given_h(h_0_p)
        h_t_p = tf.nn.sigmoid(tf.matmul(v_t, self.w) + self.hbias)

        positive_grad = tf.matmul(tf.transpose(self.x), h_0_p)
        negative_grad = tf.matmul(tf.transpose(v_t), h_t_p)

        accum_delta_w = (positive_grad - negative_grad) / tf.to_float(tf.shape(self.x)[0])
        accum_delta_vbias = tf.reduce_mean(self.x - v_t, 0)
        accum_delta_hbias = tf.reduce_mean(h_0_p - h_t_p, 0)

        delta_w_new = self.momentum*self.delta_w + self.learning_rate*(accum_delta_w-self.weight_cost*self.w)
        delta_vbias_new = self.momentum*self.delta_vbias + self.learning_rate*accum_delta_vbias
        delta_hbias_new = self.momentum*self.delta_hbias + self.learning_rate*accum_delta_hbias

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_vbias = self.delta_vbias.assign(delta_vbias_new)
        update_delta_hbias = self.delta_hbias.assign(delta_hbias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_vbias = self.vbias.assign(self.vbias + delta_vbias_new)
        update_hbias = self.hbias.assign(self.hbias + delta_hbias_new)

        self.update_deltas = [update_delta_w, update_delta_vbias, update_delta_hbias]
        self.update_weights = [update_w, update_vbias, update_hbias]

        self.compute_visible_from_hidden = self.compute_visible_func(self.y)
        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hbias)
        self.compute_visible = self.compute_visible_func(self.compute_hidden)
        self.free_energy = self.free_energy_func()
        self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self, batch_x):
        return self.sess.run(self.free_energy, feed_dict={self.x: batch_x})

    def free_energy_gap(self, x_train, x_test):
        return self.get_free_energy(x_train) - self.get_free_energy(x_test)

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas,feed_dict={self.x: batch_x})

    def fit_transform(self, X):
        return (self.fit(X)).transform(X)

    def get_weights(self):
        pass
        return self.sess.run(self.w),\
            self.sess.run(self.vbias),\
            self.sess.run(self.hbias)

    def save_weights(self, name):
        pass
        np.save(name+'_rbm_w.npy',self.sess.run(self.w))
        # Only save the hid biases for the fine-tune
        np.save(name+'_rbm_hb.npy',self.sess.run(self.hbias))

    def set_weights(self, w, vbias, hbias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.vbias.assign(vbias))
        self.sess.run(self.hbias.assign(hbias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.vbias,
                                name + '_h': self.hbias})
        saver.restore(self.sess, filename)


class GBRBM(BBRBM):
    def __init__(self,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.5,
                 weight_cost=0.0002,
                 batch_size=4,
                 n_epoches=20,
                 cd_iter=1,
                 sample_gauss_visible=False,
                 sigma=1.0):
        self.sample_gauss_visible = sample_gauss_visible
        self.sigma = sigma
        super().__init__(n_hidden,learning_rate,momentum,weight_cost,batch_size,
        n_epoches,cd_iter)

    def sample_v_given_h(self, h_0_p):
        return tf.matmul(sample_bernoulli(h_0_p), tf.transpose(self.w)) + self.vbias

    def compute_visible_func(self, hidden):
        return tf.matmul(hidden, tf.transpose(self.w)) + self.vbias

    def free_energy_func(self):
        return - tf.reduce_mean(0.5 * tf.matmul((self.x - self.vbias),
        tf.transpose(self.x - self.vbias))) \
        - tf.reduce_mean(tf.log(1 + tf.exp(tf.matmul(self.x, self.w) + self.hbias)))
