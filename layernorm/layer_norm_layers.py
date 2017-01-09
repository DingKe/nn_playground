from keras.engine import Layer, InputSpec
from keras.layers import LSTM, GRU
from keras import initializations, regularizers
from keras import backend as K

import numpy as np


def to_list(x):
    if type(x) not in [list, tuple]:
        return [x]
    else:
        return list(x)


def LN(x, gamma, beta, epsilon=1e-6, axis=-1):
    m = K.mean(x, axis=axis, keepdims=True)
    std = K.sqrt(K.var(x, axis=axis, keepdims=True) + epsilon)
    x_normed = (x - m) / (std + epsilon)
    x_normed = gamma * x_normed + beta

    return x_normed


class LayerNormalization(Layer):
    def __init__(self, axis=-1,
                 gamma_init='one', beta_init='zero',
                 gamma_regularizer=None, beta_regularizer=None,
                 weights=None, epsilon=1e-6, **kwargs): 
        self.supports_masking = True

        self.axis = to_list(axis)
        self.gamma_init = initializations.get(gamma_init)
        self.beta_init = initializations.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.initial_weights = weights

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]
        for i in self.axis:
            shape[i] = input_shape[i]
        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name))
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        return LN(x, gamma=self.gamma, beta=self.beta, 
                      axis=self.axis, epsilon=self.epsilon)

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': self.gamma_init.__name__,
                  'beta_init': self.beta_init.__name__,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormLSTM(LSTM):
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.W = self.add_weight((self.input_dim, 4 * self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)
        self.U = self.add_weight((self.output_dim, 4 * self.output_dim),
                                 initializer=self.inner_init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer)
        self.gamma_1 = self.add_weight((4 * self.output_dim,),
                                       initializer='one',
                                       name='{}_gamma_1'.format(self.name))
        self.beta_1 = self.add_weight((4 * self.output_dim,),
                                       initializer='zero',
                                       name='{}_beta_1'.format(self.name))
        self.gamma_2 = self.add_weight((4 * self.output_dim,),
                                       initializer='one',
                                       name='{}_gamma_2'.format(self.name))
        self.beta_2 = self.add_weight((4 * self.output_dim,),
                                       initializer='zero',
                                       name='{}_beta_2'.format(self.name))
        self.gamma_3 = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_gamma_3'.format(self.name))
        self.beta_3 = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_beta_3'.format(self.name))

        def b_reg(shape, name=None):
            return K.variable(np.hstack((np.zeros(self.output_dim),
                                         K.get_value(self.forget_bias_init((self.output_dim,))),
                                         np.zeros(self.output_dim),
                                         np.zeros(self.output_dim))),
                              name='{}_b'.format(self.name))
        self.b = self.add_weight((self.output_dim * 4,),
                                 initializer=b_reg,
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        z = LN(K.dot(x * B_W[0], self.W), self.gamma_1, self.beta_1) +  \
            LN(K.dot(h_tm1 * B_U[0], self.U), self.gamma_2, self.beta_2) + \
            self.b

        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim: 2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3)

        h = o * self.activation(LN(c, self.gamma_3, self.beta_3))
        return h, [h, c]
