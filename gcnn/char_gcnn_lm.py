# -*- coding: utf-8 -*-
'''Simple RNN for Language Model
'''
from __future__ import print_function
import os

from keras.models import Model
from keras.layers import Input, Embedding, Dense, TimeDistributed
from keras.optimizers import *

from gcnn import GCNN
from char_generator import TextLoader


def LM(batch_size, window_size=3, vocsize=20000, embed_dim=20, hidden_dim=30, nb_layers=1):
    x = Input(batch_shape=(batch_size, None))
    # mebedding
    y = Embedding(vocsize+2, embed_dim, mask_zero=False)(x)
    for i in range(nb_layers-1):
        y = GCNN(hidden_dim, window_size=window_size,
                 name='gcnn{}'.format(i + 1))(y)
    y = GCNN(hidden_dim, window_size=window_size, 
             name='gcnn{}'.format(nb_layers))(y)
    y = TimeDistributed(Dense(vocsize+2, activation='softmax', name='dense{}'.format(nb_layers)))(y)

    model = Model(input=x, output=y)

    return model


def run_demo():
    batch_size = 50 
    nb_epoch = 100 
    nb_layers = 3

    max_len = 50 
    window_size = 5
    
    # Prepare data
    path = './data/tinyshakespeare'
    data_loader = TextLoader(path, batch_size, max_len)
    vocsize = data_loader.vocab_size
    print('vocsize: {}'.format(vocsize))

    # Build model
    model = LM(batch_size, window_size=window_size, vocsize=vocsize, nb_layers=nb_layers)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')

    
    train_samples = data_loader.num_batches * batch_size

    # Start training
    model.summary()   
    model.fit_generator(data_loader(), samples_per_epoch=train_samples,                         
                            nb_epoch=nb_epoch, verbose=1)


if __name__ == '__main__':
    run_demo()
