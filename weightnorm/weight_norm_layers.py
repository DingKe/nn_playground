from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense, Conv2D, GRU


class WeightNormDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.g = self.add_weight(shape=(self.units,),
                                 initializer='one',
                                 name='g')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        kernel = self.kernel * self.g / K.sqrt(K.sum(K.square(self.kernel), axis=0))
        output = K.dot(inputs, kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class WeightNormConv2D(Conv2D):
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.g = self.add_weight(shape=(1, 1, 1, self.filters),
                                 initializer='one',
                                 name='g')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})

        self.built = True

    def call(self, x):
        kernel = self.kernel * self.g / K.sqrt(K.sum(K.square(self.kernel), axis=[0, 1, 2], keepdims=True))
        output = K.conv2d(x, kernel, strides=self.strides,
                          padding=self.padding,
                          data_format=self.data_format)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)
        if self.activation is not None:
            output = self.activation(output)
        return output


class WeightNormGRU(GRU):
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        self.g_kernel_z = self.add_weight((self.units,),
                                   initializer='one',
                                   name='g_kernel_z')
        self.g_kernel_r = self.add_weight((self.units,),
                                   initializer='one',
                                   name='g_kernel_r')
        self.g_kernel_h = self.add_weight((self.units,),
                                   initializer='one',
                                   name='g_kernel_h')
        self.g_recurrent_kernel_z = self.add_weight((self.units,),
                                   initializer='one',
                                   name='g_recurrent_kernel_z')
        self.g_recurrent_kernel_r = self.add_weight((self.units,),
                                   initializer='one',
                                   name='g_recurrent_kernel_r')
        self.g_recurrent_kernel_h = self.add_weight((self.units,),
                                   initializer='one',
                                   name='g_recurrrent_kernel_h')
        self.built = True

    def preprocess_input(self, x, training=None):
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        kernel_z = self.kernel_z * self.g_kernel_z / K.sqrt(K.sum(K.square(self.g_kernel_z), axis=0))
        kernel_r = self.kernel_r * self.g_kernel_r / K.sqrt(K.sum(K.square(self.g_kernel_r), axis=0))
        kernel_h = self.kernel_h * self.g_kernel_h / K.sqrt(K.sum(K.square(self.g_kernel_h), axis=0))
        recurrent_kernel_z = self.recurrent_kernel_z * self.g_recurrent_kernel_z / K.sqrt(K.sum(K.square(self.g_recurrent_kernel_z), axis=0))
        recurrent_kernel_r = self.recurrent_kernel_r * self.g_recurrent_kernel_r / K.sqrt(K.sum(K.square(self.g_recurrent_kernel_r), axis=0))
        recurrent_kernel_h = self.recurrent_kernel_h * self.g_recurrent_kernel_h / K.sqrt(K.sum(K.square(self.g_recurrent_kernel_h), axis=0))

        x_z = K.dot(x * B_W[0], kernel_z)
        x_r = K.dot(x * B_W[1], kernel_r)
        x_h = K.dot(x * B_W[2], kernel_h)
        if self.use_bias:
            x_z += self.bias_z
            x_r += self.bias_r
            x_h += self.bias_h

        z = self.recurrent_activation(x_z + K.dot(h_tm1 * B_U[0], recurrent_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(h_tm1 * B_U[1], recurrent_kernel_r))
        hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], recurrent_kernel_h))
        h = z * h_tm1 + (1 - z) * hh

        return h, [h]


# Aliases

WeightNormConvolution2D = WeightNormConv2D
