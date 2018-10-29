import tensorflow as tf
import numpy as np
from rbm_tf import BBRBM, GBRBM
from sklearn.metrics import mean_squared_error

class DBN():
    def __init__(self,
                 hidden_layers_structure=[512, 512, 512],
                 # activation_function='sigmoid',
                 # optimization_algorithm='sgd',
                 n_epoches=10,
                 batch_size=20,
                 pretrain=False,
                 learning_rate_rbm=[1e-3,1e-2,1e-2],
                 weight_cost=0.0002,
                 # momentum=0.95,
                 rbm_gauss_visible=False,
                 sample_gauss_visible=False,
                 sigma=1,
                 ):
        self.hidden_layers_structure = hidden_layers_structure
        # self.pretrain = pretrain
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.learning_rate_rbm = learning_rate_rbm
        self.weight_cost = weight_cost
        self.rbm_gauss_visible = rbm_gauss_visible
        self.sample_gauss_visible = sample_gauss_visible
        self.sigma = sigma
        self.rbm_numbers = len(self.hidden_layers_structure)
        self.rbm_layers = []
        for l, n_hidden_units in enumerate(self.hidden_layers_structure):
            if l == 0:
                if self.rbm_gauss_visible:
                    self.rbm_layers.append(
                        GBRBM(
                            n_hidden=n_hidden_units,
                            weight_cost=self.weight_cost,
                            batch_size=self.batch_size,
                            n_epoches=self.n_epoches,
                            learning_rate=(self.learning_rate_rbm)[l],
                            sample_gauss_visible=self.sample_gauss_visible,
                            sigma=self.sigma))
                else:
                    self.rbm_layers.append(
                        BBRBM(
                            n_hidden=n_hidden_units,
                            weight_cost=self.weight_cost,
                            batch_size=self.batch_size,
                            n_epoches=self.n_epoches,
                            learning_rate=(self.learning_rate_rbm)[l]))
            else:
                self.rbm_layers.append(
                    BBRBM(
                        n_hidden=n_hidden_units,
                        weight_cost=self.weight_cost,
                        batch_size=self.batch_size,
                        n_epoches=self.n_epoches,
                        learning_rate=(self.learning_rate_rbm)[l]))

    def fit(self, X, X_vali = None):
        input_data = np.array(X)
        input_vali = X_vali
        for rbm in self.rbm_layers:
            rbm.fit(input_data, input_vali)
            input_data = rbm.transform(input_data)
            if input_vali is not None:
                input_vali = rbm.transform(input_vali)
        return self

    def transform(self, X):
        input_data = X
        for rbm in self.rbm_layers:
            input_data = rbm.transform(input_data)
        return input_data

    def transform_inv(self, y):
        pass
        input_data = y
        for rbm in reversed(self.rbm_layers):
            input_data = rbm.transform_inv(input_data)
        return input_data

    def get_hidden_samples(self, X, number):
        input_data = X
        for i in range(number):
            input_data = (self.rbm_layers)[i].transform(input_data)
        return input_data

    def reconstruct(self, X):
        input_data = X
        for rbm in self.rbm_layers:
            input_data = rbm.transform(input_data)
        for rbm in reversed(self.rbm_layers):
            input_data = rbm.transform_inv(input_data)
        return input_data

    def get_err(self, X):
        return mean_squared_error(X, self.reconstruct(X))

    def get_free_energy(self, X):
        return (self.rbm_layers)[0].get_free_energy(X)

    def free_energy_gap(self, x_train, x_test):
        input_train = x_train
        input_test = x_test
        gap = []
        for rbm in self.rbm_layers:
            gap.append(rbm.free_energy_gap(input_train, input_test))
            input_train = rbm.transform(input_train)
            input_test = rbm.transform(input_test)

        return gap

    def get_weights(self):
        w = []
        vb = []
        hb = []
        for rbm in self.rbm_layers:
            a, b, c = rbm.get_weights()
            w.append(a)
            vb.append(b)
            hb.append(c)
        return w, vb, hb

    def save_weights(self, name):
        for i in range(self.rbm_numbers):
            w, v, h = (self.rbm_layers)[i].get_weights()
            np.save(name+'_rbm_{0}_w.npy'.format(i+1),w)
            # Only save the hid biases for the fine-tune
            np.save(name+'_rbm_{0}_hb.npy'.format(i+1),h)

    def set_weights(self, w, visible_bias, hidden_bias):
        for i in range(self.rbm_numbers):
            self.rbm_layers[i].set_weights(w[i], visible_bias[i],
                                            hidden_bias[i])
