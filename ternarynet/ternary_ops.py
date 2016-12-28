# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K


def select(condition, t, e):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.select(condition, t, e)
    elif K.backend() == 'theano':
        import theano.tensor as tt
        return tt.switch(condition, t, e)


def ternarize(W, H=1):
    '''The weights' binarization function, 

    # References:
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902)
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    W /= H

    ones = K.ones_like(W)
    zeros = K.zeros_like(W)
    Wt = select(W > 0.5, ones, select(W <= -0.5, -ones, zeros))

    Wt *= H

    return W + K.stop_gradient(Wt - W)
