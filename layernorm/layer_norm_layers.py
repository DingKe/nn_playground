from keras.engine import Layer, InputSpec
from keras.layers import LSTM, GRU
from keras import initializers, regularizers
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
                 epsilon=1e-6, **kwargs): 
        super(LayerNormalization, self).__init__(**kwargs)

        self.axis = to_list(axis)
        self.gamma_init = initializers.get(gamma_init)
        self.beta_init = initializers.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]
        for i in self.axis:
            shape[i] = input_shape[i]
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs, mask=None):
        return LN(inputs, gamma=self.gamma, beta=self.beta, 
                  axis=self.axis, epsilon=self.epsilon)

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': initializers.serialize(self.gamma_init),
                  'beta_init': initializers.serialize(self.beta_init),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.gamma_regularizer)}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormLSTM(LSTM):
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = [InputSpec(shape=(batch_size, self.units)),
                           InputSpec(shape=(batch_size, self.units))]


        # initial states: 2 all-zero tensors of shape (units)
        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, 4 * self.units),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer=self.recurrent_initializer,
                                regularizer=self.recurrent_regularizer,
                                constraint=self.recurrent_constraint)

        self.gamma_1 = self.add_weight(shape=(4 * self.units,),
                                       initializer='one',
                                       name='gamma_1')
        self.beta_1 = self.add_weight(shape=(4 * self.units,),
                                      initializer='zero',
                                      name='beta_1')
        self.gamma_2 = self.add_weight(shape=(4 * self.units,),
                                       initializer='one',
                                       name='gamma_2')
        self.beta_2 = self.add_weight(shape=(4 * self.units,),
                                      initializer='zero',
                                      name='beta_2')
        self.gamma_3 = self.add_weight(shape=(self.units,),
                                       initializer='one',
                                       name='gamma_3')
        self.beta_3 = self.add_weight((self.units,),
                                       initializer='zero',
                                       name='beta_3')

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        return inputs 

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        z = LN(K.dot(x * B_W[0], self.kernel), self.gamma_1, self.beta_1) +  \
            LN(K.dot(h_tm1 * B_U[0], self.recurrent_kernel), self.gamma_2, self.beta_2)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(LN(c, self.gamma_3, self.beta_3))
        return h, [h, c]
