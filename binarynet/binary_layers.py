# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K

from keras.layers import InputSpec, Layer, Dense, Convolution2D
from keras import constraints
from keras import initializations

from binary_ops import binarize


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"name": self.__call__.__name__,
                "min_value": self.min_value,
                "max_value": self.max_value}


class BinaryDense(Dense):
    ''' Binarized Dense layer
    References: 
    "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, output_dim, H=1., W_lr_multiplier='Glorot', b_lr_multiplier=None, **kwargs):
        self.H = H
        self.W_lr_multiplier = W_lr_multiplier
        self.b_lr_multiplier = b_lr_multiplier
        
        super(BinaryDense, self).__init__(output_dim, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.output_dim)))
            #print('Glorot H: {}'.format(self.H))
            
        if self.W_lr_multiplier == 'Glorot':
            self.W_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.output_dim)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))
            
        self.W_constraint = Clip(-self.H, self.H)
        
        self.init = initializations.get('uniform')
        self.init_func = lambda shape, name: self.init(shape, scale=self.H, name=name)
        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init_func,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.lr_multipliers = [self.W_lr_multiplier, self.b_lr_multiplier]
        else:
            self.lr_multipliers = [self.W_lr_multiplier]

        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        Wb = binarize(self.W, H=self.H)

        if self.bias:
            output = self.activation(K.dot(x, Wb) + self.b)
        else:
            output = self.activation(K.dot(x, Wb))
        return output
        
    def get_config(self):
        config = {'H': self.H,
                  'W_lr_multiplier': self.W_lr_multiplier,
                  'b_lr_multiplier': self.b_lr_multiplier}
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryConvolution2D(Convolution2D):
    '''Binarized Convolution2D layer
    References: 
    "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, nb_filter, nb_row, nb_col, W_lr_multiplier='Glorot', 
                 b_lr_multiplier=None, H=1., **kwargs):
        self.H = H
        self.W_lr_multiplier = W_lr_multiplier
        self.b_lr_multiplier = b_lr_multiplier
        
        super(BinaryConvolution2D, self).__init__(nb_filter, nb_row, nb_col, **kwargs)
        
    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
            
        if self.H == 'Glorot':
            nb_input = int(stack_size * self.nb_row * self.nb_col)
            nb_output = int(self.nb_filter * self.nb_row * self.nb_col)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            #print('Glorot H: {}'.format(self.H))
            
        if self.W_lr_multiplier == 'Glorot':
            nb_input = int(stack_size * self.nb_row * self.nb_col)
            nb_output = int(self.nb_filter *self.nb_row * self.nb_col)
            self.W_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))
            
        self.init = initializations.get('uniform')
        self.init_func = lambda shape, name: self.init(shape, scale=self.H, name=name)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init_func,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.lr_multipliers = [self.W_lr_multiplier, self.b_lr_multiplier]
        else:
            self.lr_multipliers = [self.W_lr_multiplier]
        
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)


        W_constraint = Clip(-self.H, self.H)
        self.constraints[self.W] = W_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        Wb = binarize(self.W, H=self.H) 
        conv_out = K.conv2d(x, Wb, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)
                                
        if self.bias:
            if self.dim_ordering == 'th':
                conv_out = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                conv_out = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
                
        output = self.activation(conv_out)
        return output
        
    def get_config(self):
        config = {'H': self.H,
                  'W_lr_multiplier': self.W_lr_multiplier,
                  'b_lr_multiplier': self.b_lr_multiplier}
        base_config = super(BinaryConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
