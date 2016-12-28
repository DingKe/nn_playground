# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K

from keras.layers import InputSpec, Layer, Dense, Convolution2D
from keras import constraints
from keras import initializations

from ternary_ops import ternarize


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


class TernaryDense(Dense):
    ''' Ternarized Dense layer

    References: 
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902}
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    def __init__(self, output_dim, H=1., W_lr_multiplier='Glorot', b_lr_multiplier=None, **kwargs):
        self.H = H
        self.W_lr_multiplier = W_lr_multiplier
        self.b_lr_multiplier = b_lr_multiplier
        
        super(TernaryDense, self).__init__(output_dim, **kwargs)
    
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
        Wb = ternarize(self.W, H=self.H)

        if self.bias:
            output = self.activation(K.dot(x, Wb) + self.b)
        else:
            output = self.activation(K.dot(x, Wb))
        return output
        
    def get_config(self):
        config = {'H': self.H,
                  'W_lr_multiplier': self.W_lr_multiplier,
                  'b_lr_multiplier': self.b_lr_multiplier}
        base_config = super(TernayDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
