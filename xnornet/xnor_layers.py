# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from binary_ops import xnorize

from binary_layers import BinaryDense, BinaryConv2D


class XnorDense(BinaryDense):
    '''XNOR Dense layer
    References: 
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](http://arxiv.org/abs/1603.05279)
    '''
    def call(self, inputs, mask=None):
        inputs_a, inputs_b = xnorize(inputs, 1., axis=1, keepdims=True) # (nb_sample, 1)
        kernel_a, kernel_b = xnorize(self.kernel, self.H, axis=0, keepdims=True) # (1, units)
        output = K.dot(inputs_b, kernel_b) * kernel_a * inputs_a
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class XnorConv2D(BinaryConv2D):
    '''XNOR Conv2D layer
    References: 
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](http://arxiv.org/abs/1603.05279)
    '''
    def call(self, inputs):
        _, kernel_b = xnorize(self.kernel, self.H)
        _, inputs_b = xnorize(inputs)
        outputs = K.conv2d(inputs_b, kernel_b, strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        # calculate Wa and xa
        
        # kernel_a
        mask = K.reshape(self.kernel, (-1, self.filters)) # self.nb_row * self.nb_col * channels, filters 
        kernel_a = K.stop_gradient(K.mean(K.abs(mask), axis=0)) # filters
        
        # inputs_a
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        mask = K.mean(K.abs(inputs), axis=channel_axis, keepdims=True) 
        ones = K.ones(self.kernel_size + (1, 1))
        inputs_a = K.conv2d(mask, ones, strides=self.strides,
                      padding=self.padding,
                      data_format=self.data_format,
                      dilation_rate=self.dilation_rate) # nb_sample, 1, new_nb_row, new_nb_col
        if self.data_format == 'channels_first':
            outputs = outputs * K.stop_gradient(inputs_a) * K.expand_dims(K.expand_dims(K.expand_dims(kernel_a, 0), -1), -1)
        else:
            outputs = outputs * K.stop_gradient(inputs_a) * K.expand_dims(K.expand_dims(K.expand_dims(kernel_a, 0), 0), 0)
                                
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# Aliases

XnorConvolution2D = XnorConv2D
