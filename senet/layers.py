# -*- coding: utf-8 -*-
from keras import backend as K
from keras import initializers, constraints, regularizers
from keras.layers import InputSpec, Layer

from keras.utils import conv_utils


class SE(Layer):
    def __init__(self, 
                 ratio, 
                 data_format=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SE, self).__init__(**kwargs)

        self.ratio = ratio
        self.data_format= conv_utils.normalize_data_format(data_format)

        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 4
        self.input_spec = InputSpec(shape=input_shape)

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3  
        channels = input_shape[channel_axis]

        self.kernel1 = self.add_weight(shape=(channels, channels // self.ratio),
                                      initializer=self.kernel_initializer,
                                      name='kernel1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias1 = self.add_weight(shape=(channels // self.ratio,),
                                         initializer=self.bias_initializer,
                                         name='bias1',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias1 = None

        self.kernel2 = self.add_weight(shape=(channels // self.ratio, channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias2 = self.add_weight(shape=(channels,),
                                         initializer=self.bias_initializer,
                                         name='bias2',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
        else:
            self.bias2 = None

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if self.data_format == 'channels_first':
            sq = K.mean(inputs, [2, 3])
        else:
            sq = K.mean(inputs, [1, 2])

        ex = K.dot(sq, self.kernel1)
        if self.use_bias:
            ex = K.bias_add(ex, self.bias1)
        ex= K.relu(ex)

        ex = K.dot(ex, self.kernel2)
        if self.use_bias:
            ex = K.bias_add(ex, self.bias2)
        ex= K.sigmoid(ex)

        if self.data_format == 'channels_first':
            ex = K.expand_dims(ex, -1)
            ex = K.expand_dims(ex, -1)
        else:
            ex = K.expand_dims(ex, 1)
            ex = K.expand_dims(ex, 1)

        return inputs * ex

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'data_format': self.data_format,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(SE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
