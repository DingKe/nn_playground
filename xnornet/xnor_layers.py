# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from binary_ops import xnorize

from binary_layers import BinaryDense, BinaryConvolution2D


class XnorDense(BinaryDense):
    '''XNOR Dense layer
    References: 
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](http://arxiv.org/abs/1603.05279)
    '''
    def call(self, x, mask=None):
        xa, xb = xnorize(x, 1., axis=1, keepdims=True) # (nb_sample, 1)
        Wa, Wb = xnorize(self.W, self.H, axis=0, keepdims=True) # (1, output_dim)
        out = K.dot(xb, Wb) * Wa * xa
        if self.bias:
            out += self.b
        out = self.activation(out)
            
        return out


class XnorConvolution2D(BinaryConvolution2D):
    '''XNOR Conv2D layer
    References: 
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](http://arxiv.org/abs/1603.05279)
    '''
    def call(self, x, mask=None):
        _, Wb = xnorize(self.W, self.H)
        _, xb = xnorize(x)
        conv_out = K.conv2d(xb, Wb, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)
                            
        if self.dim_ordering == 'th':
            # calculate Wa and xa
            
            # Wa
            mask = K.reshape(self.W, (self.nb_filter, -1)) # nb_filter, stack_size * self.nb_row * self.nb_col
            Wa = K.stop_gradient(K.mean(K.abs(mask), axis=1)) # nb_filter
            
            # xa
            mask = K.permute_dimensions(x, (0, 2, 3, 1)) # nb_sample, nb_row, nb_col, stack_size
            mask = K.mean(K.abs(mask), axis=-1, keepdims=True) # nb_sample, nb_row, nb_col, 1
            mask = K.permute_dimensions(mask, (0, 3, 1, 2)) # nb_sample, 1, nb_row, nb_col
            xa = K.conv2d(mask, K.ones((1, 1, self.nb_row, self.nb_col)), strides=self.subsample,
                        border_mode=self.border_mode,
                        dim_ordering=self.dim_ordering) # nb_sample, 1, new_nb_row, new_nb_col
            conv_out = conv_out * K.stop_gradient(xa) * K.expand_dims(K.expand_dims(K.expand_dims(Wa, 0), -1), -1)
        elif self.dim_ordering == 'tf':
            # calculate xa
            
            # Wa
            mask = K.reshape(self.W, (-1, self.nb_filter)) # stack_size * self.nb_row * self.nb_col, nb_filter
            Wa = K.stop_gradient(K.mean(K.abs(mask), axis=0)) # nb_filter
            
            # xa
            mask = K.mean(K.abs(mask), axis=-1, keepdims=True) # nb_sample, nb_row, nb_col, 1
            xa = K.conv2d(mask, K.ones((self.nb_row, self.nb_col, 1, 1)), strides=self.subsample,
                        border_mode=self.border_mode,
                        dim_ordering=self.dim_ordering) # nb_sample, new_nb_row, new_nb_col, 1
            conv_out = conv_out * K.stop_gradient(xa) * K.expand_dims(K.expand_dims(K.expand_dims(Wa, 0), 0), 0)
                                
        if self.bias:
            if self.dim_ordering == 'th':
                conv_out = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                conv_out = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering: ' + self.dim_ordering)
                
        output = self.activation(conv_out)
        return output


# Aliases

XnorConv2D = XnorConvolution2D
