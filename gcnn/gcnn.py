# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.layers import Layer, InputSpec

from keras.utils.np_utils import conv_output_length

class GCNN(Layer):
    '''Gated Convolutional Networks

    # Arguments

    # References
        - Dauphin et al. [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
    '''
    def __init__(self, output_dim, window_size=3, subsample_length=1,
                 init='uniform', activation='linear',
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, 
                 weights=None, bias=True, 
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.window_size = window_size
        self.subsample = (subsample_length, 1)

        self.bias = bias
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights

        self.supports_masking = False 
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(GCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.input_spec = [InputSpec(shape=input_shape)]
        self.W_shape = (self.window_size, 1, input_dim, self.output_dim * 2)

        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

        if self.bias:
            self.b = K.zeros((self.output_dim * 2,), name='{}_b'.format(self.name))
            self.trainable_weights += [self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size,
                                        'valid',
                                        self.subsample[0])
        return (input_shape[0], length, self.output_dim)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape

        if self.window_size > 1:
            x = K.asymmetric_temporal_padding(x, self.window_size-1, 0)
        x = K.expand_dims(x, 2)  # add a dummy dimension

        # z, g
        output = K.conv2d(x, self.W, strides=self.subsample,
                          border_mode='valid',
                          dim_ordering='tf')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.bias:
            output += K.reshape(self.b, (1, 1, self.output_dim * 2))
        z  = output[:, :, :self.output_dim]
        g = output[:, :, self.output_dim:]

        return self.activation(z) * K.sigmoid(g)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'window_size': self.window_size,
                  'init': self.init.__name__,
                  'subsample_length': self.subsample[0],
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(GCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
