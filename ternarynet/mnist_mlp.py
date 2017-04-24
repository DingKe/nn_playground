'''Trains a simple binarize fully connected NN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.10% test accuracy after 20 epochs using theano backend
'''


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from ternary_ops import ternarize
from ternary_layers import TernaryDense


class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    '''
    def call(self, inputs, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            inputs = K.in_train_phase(
                        K.dropout(inputs, self.rate, noise_shape) * (1. - self.rate), 
                        inputs)# multiplied by (1. - self.rate) for compensation
        return inputs 


def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)


batch_size = 100
epochs = 20
nb_classes = 10

H = 'Glorot'
kernel_lr_multiplier = 'Glorot'

# network
num_unit = 2048
num_hidden = 3
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
drop_in = 0.2
drop_hidden = 0.5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

model = Sequential()
model.add(DropoutNoScale(drop_in, input_shape=(784,), name='drop0'))
for i in range(num_hidden):
    model.add(TernaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
              name='dense{}'.format(i+1)))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
    model.add(Activation(ternary_tanh, name='act{}'.format(i+1)))
    model.add(DropoutNoScale(drop_hidden, name='drop{}'.format(i+1)))
model.add(TernaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,
          name='dense'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))

model.summary()

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
