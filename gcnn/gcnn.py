# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec

from keras.utils.conv_utils import conv_output_length

class GCNN(Layer):
    '''Gated Convolutional Networks

    # Arguments

    # References
        - Dauphin et al. [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
    '''
    def __init__(self, output_dim, window_size=3, stride=1,
                 kernel_initializer='uniform', bias_initializer='zero',
                 activation='linear', activity_regularizer=None,
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, 
                 use_bias=True, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(GCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(shape=input_shape)]
        self.kernel_shape = (self.window_size, 1, input_dim, self.output_dim * 2)
        self.kernel = self.add_weight(self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.output_dim * 2,),
                                        initializer=self.bias_initializer,
                                        name='b',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size, 'valid',
                                        self.strides[0])
        return (input_shape[0], length, self.output_dim)

    def call(self, x):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape

        if self.window_size > 1:
            x = K.temporal_padding(x, (self.window_size-1, 0))
        x = K.expand_dims(x, 2)  # add a dummy dimension

        # z, g
        output = K.conv2d(x, self.kernel, strides=self.strides,
                          padding='valid',
                          data_format='channels_last')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        z  = output[:, :, :self.output_dim]
        g = output[:, :, self.output_dim:]

        return self.activation(z) * K.sigmoid(g)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'window_size': self.window_size,
                  'init': self.init.get_config(),
                  'stride': self.strides[0],
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activy_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'use_bias': self.use_bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(GCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
