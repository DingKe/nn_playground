# -*- coding: utf-8 -*-
'''Simple RNN for Language Model
'''
from __future__ import print_function
import os

from keras.models import Model
from keras.layers import Input, Embedding, Dense, TimeDistributed
from keras.optimizers import *

from ternary_layers import *
from imdb_generator import IMDBLM


def LM(batch_size, window_size=3, vocsize=20000, embed_dim=20, hidden_dim=30, nb_layers=1):
    x = Input(batch_shape=(batch_size, None))
    # mebedding
    y = Embedding(vocsize+2, embed_dim, mask_zero=False)(x)
    for i in range(nb_layers-1):
        y = TernaryRNN(hidden_dim, return_sequences=True, name='trnn{}'.format(i + 1))(y)
    y = TernaryRNN(hidden_dim, return_sequences=True, name='trnn{}'.format(nb_layers))(y)
    y = TimeDistributed(Dense(vocsize+2, activation='softmax', name='dense{}'.format(nb_layers)))(y)

    model = Model(input=x, output=y)

    return model


def train_model():
    batch_size = 32 
    epochs = 100 

    vocsize = 2000 # top 2k
    max_len = 30 
    train_ratio = 0.99

    # Build model
    model = LM(batch_size, vocsize=vocsize, nb_layers=1)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')

    # Prepare data
    path = './data/imdb-full.pkl'
    # Train
    train_gen = IMDBLM(path=path, max_len=max_len, vocab_size=vocsize, shuffle=True,
                       which_set='train', train_ratio=train_ratio, batch_size=batch_size)
    # Validation 
    val_gen = IMDBLM(path=path, max_len=max_len, vocab_size=vocsize,
                     which_set='validation', train_ratio=train_ratio, batch_size=batch_size)

    train_steps = 2000
    val_steps = 200

    # Start training
    model.summary()
    model.fit_generator(train_gen(), steps_per_epoch=train_steps, 
                        validation_data=val_gen(), validation_steps=val_steps,
                        epochs=epochs, verbose=1)


def run_demo():
    train_model()


if __name__ == '__main__':
    run_demo()
