# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K


def switch(condition, t, e):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.where(condition, t, e)
    elif K.backend() == 'theano':
        import theano.tensor as tt
        return tt.switch(condition, t, e)


def _ternarize(W, H=1):
    '''The weights' ternarization function, 

    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    W /= H

    ones = K.ones_like(W)
    zeros = K.zeros_like(W)
    Wt = switch(W > 0.5, ones, switch(W <= -0.5, -ones, zeros))

    Wt *= H

    return Wt


def ternarize(W, H=1):
    '''The weights' ternarization function, 

    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    Wt = _ternarize(W, H)
    return W + K.stop_gradient(Wt - W)


def ternarize_dot(x, W):
    '''For RNN (maybe Dense or Conv too). 
    Refer to 'Recurrent Neural Networks with Limited Numerical Precision' Section 3.1
    '''
    Wt = _ternarize(W)
    return K.dot(x, W) + K.stop_gradient(K.dot(x, Wt - W))
