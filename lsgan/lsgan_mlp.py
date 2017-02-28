#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a Least Square  Generative Adversarial Network (WGAN) on the MNIST
"""
from __future__ import print_function
from PIL import Image
from six.moves import range

import keras.backend as K
K.set_image_dim_ordering('th')

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils.generic_utils import Progbar

import numpy as np
np.random.seed(1337)


def build_generator(latent_size):
    model = Sequential()
    model.add(Dense(1024, input_dim=latent_size, activation='relu'))
    model.add(Dense(28 * 28, activation='tanh'))
    model.add(Reshape((1, 28, 28)))

    return model 


def build_discriminator():
    f = Sequential()
    f.add(Flatten(input_shape=(1, 28, 28)))
    f.add(Dense(256))
    f.add(Activation('relu'))
    f.add(Dense(128))
    f.add(Activation('relu'))
    f.add(Dense(1, activation='linear'))

    image = Input(shape=(1, 28, 28))
    score = f(image)

    model = Model(image, score)

    return model


if __name__ == '__main__':

    nb_epochs = 5000
    batch_size = 50
    latent_size = 20

    lr = 0.001

    # build the discriminator
    disc = build_discriminator()
    disc.compile(
        optimizer=Adam(lr=lr),
        loss='mse'
    )

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, ))
    # get a fake image
    fake = generator(latent)
    # we only want to be able to train generation for the combined model
    disc.trainable = False
    fake = disc(fake)
    combined = Model(input=latent, output=fake)
    combined.compile(
        optimizer=Adam(lr=lr),
        loss='mse'
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_disc_loss = []
        epoch_gen_loss = []

        index = 0
        while index < nb_batches:
            ## discrminator
            progress_bar.update(index)
            index += 1

            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))
            # generate a batch of fake images
            generated_images = generator.predict(noise, verbose=0)

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            X = np.concatenate((image_batch, generated_images))
            # a = -1, b = 1
            y = np.array([1] * len(image_batch) + [-1] * batch_size)

            epoch_disc_loss.append(disc.train_on_batch(X, y))


            ## generator
            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))
            target = np.ones(batch_size) # c == b == 1
            epoch_gen_loss.append(combined.train_on_batch(noise, target))

        print('\n[Loss_D: {:.3f}, Loss_G: {:.3f}]'.format(np.mean(epoch_disc_loss), np.mean(epoch_gen_loss)))

        # save weights every epoch
        if False:
            generator.save_weights(
                'mlp_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
            disc.save_weights(
                'mlp_disc_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))
        # get a batch to display
        generated_images = generator.predict(noise, verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'mlp_epoch_{0:03d}_generated.png'.format(epoch))
