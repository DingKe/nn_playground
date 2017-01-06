from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense, Conv2D, GRU


class WeightNormDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.g = self.add_weight((self.output_dim,),
                                 initializer=self.init,
                                 name='{}_g'.format(self.name))
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        W = self.W * self.g / K.sqrt(K.sum(K.square(self.W), axis=0))
        output = K.dot(x, W)
        if self.bias:
            output += self.b
        return self.activation(output)


class WeightNormConv2D(Conv2D):
    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
            self.g_shape = (self.nb_filter, 1, 1, 1)
            self.reduce_axes = [1, 2, 3]
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
            self.g_shape = (1, 1, 1, self.nb_filter)
            self.reduce_axes = [0, 1, 2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.g = self.add_weight(self.g_shape,
                                 initializer='one',
                                 name='{}_g'.format(self.name))
        if self.bias:
            self.b = self.add_weight((self.nb_filter,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        W = self.W * self.g / K.sqrt(K.sum(K.square(self.W), axis=self.reduce_axes, keepdims=True))
        output = K.conv2d(x, W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output


class WeightNormGRU(GRU):
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

            self.W_z = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_z'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_z = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_z'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_z = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_z'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_r = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_r'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_r = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_r'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_r = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_r'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_h = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_h'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_h = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_h'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_h = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_h'.format(self.name),
                                       regularizer=self.b_regularizer)

            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

            self.g_W_z = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_g_W_z'.format(self.name))
            self.g_W_r = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_g_W_r'.format(self.name))
            self.g_W_h = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_g_W_h'.format(self.name))
            self.g_U_z = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_g_U_z'.format(self.name))
            self.g_U_r = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_g_U_r'.format(self.name))
            self.g_U_h = self.add_weight((self.output_dim,),
                                       initializer='one',
                                       name='{}_g_U_h'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        W_z = self.W_z * self.g_W_z / K.sqrt(K.sum(K.square(self.g_W_z), axis=0))
        W_r = self.W_r * self.g_W_r / K.sqrt(K.sum(K.square(self.g_W_r), axis=0))
        W_h = self.W_h * self.g_W_h / K.sqrt(K.sum(K.square(self.g_W_h), axis=0))
        U_z = self.U_z * self.g_U_z / K.sqrt(K.sum(K.square(self.g_U_z), axis=0))
        U_r = self.U_r * self.g_U_r / K.sqrt(K.sum(K.square(self.g_U_r), axis=0))
        U_h = self.U_h * self.g_U_h / K.sqrt(K.sum(K.square(self.g_U_h), axis=0))

        x_z = K.dot(x * B_W[0], W_z) + self.b_z
        x_r = K.dot(x * B_W[1], W_r) + self.b_r
        x_h = K.dot(x * B_W[2], W_h) + self.b_h

        z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], U_z))
        r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], U_r))
        hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], U_h))
        h = z * h_tm1 + (1 - z) * hh

        return h, [h]


# Aliases

WeightNormConvolution2D = WeightNormConv2D
