'''This script demonstrates how to build a adversarial variational bayes with Keras.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Activation, Concatenate, Dot
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import *

np.random.seed(1111)  # for reproducibility

training = True 

batch_size = 50
n = 784 # for datapoint
m = 2 # for hidden variables
l = 5 # for random noise
hidden_dim = 256
epochs = 50
epsilon_std = 1.0
loss = 'categorical_crossentorpy' # 'mse' or 'categorical_crossentropy'

decay = 1e-4 # weight decay, a.k. l2 regularization
use_bias = True

## Encoder
x = Input(shape=(n,))
g = Input(shape=(l,))
x_g = Concatenate(-1)([x, g]) 
h_encoded = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='relu')(x_g)
z = Dense(m, activation='relu', kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)(h_encoded)

encoder = Model([x, g], z)

## Decoder
decoder_h = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='relu')
decoder_mean = Dense(n, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='sigmoid')
h_decoded = decoder_h(z)
x_hat = decoder_mean(h_decoded)

rec = Model([x, g], x_hat)
rec.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')

## Discriminator
x_h = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='relu')
z_h = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='relu')

### gan_g
x_h.trainable = False
z_h.trainable = False
T = Dot(-1)([x_h(x), z_h(z)])
gan_g = Model([x, g], T)
gan_g.compile(optimizer=Adam(1e-4), loss=lambda y_true, y_predict: -K.mean(y_predict, -1))

### gan_d
x_h.trainable = True 
z_h.trainable = True
x = Input(shape=(n,))
z = Input(shape=(m,))
fake = Dot(-1)([x_h(x), z_h(z)])
fake = Activation('sigmoid')(fake)
gan_d = Model([x, z], fake)
gan_d.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

if training:
    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    ids = range(len(x_train))
    for i in range(epochs):
        print('Epoch {}:'.format(i + 1))
        rec_loss = [] 
        gen_loss = [] 
        dis_loss = [] 
        for s in range(0, len(ids), batch_size):
            x = x_train[ids[s:s+batch_size]]
            y = y_train[ids[s:s+batch_size]]
            bs = len(x)

            eps = np.random.randn(bs, l)
            z = np.random.randn(bs, m)

            # reconstruction 
            loss = rec.train_on_batch([x, eps], x)
            rec_loss.append(loss)

            # encoder
            loss = gan_g.train_on_batch([x, eps], [1] * bs)
            gen_loss.append(loss)

            # discriminator
            z_fake = encoder.predict([x, eps])
            x = np.concatenate([x, x], axis=0)
            z = np.concatenate([z, z_fake], axis=0)
            y = np.asarray([1] * bs + [0] * bs, dtype='float32')

            loss = gan_d.train_on_batch([x, z], y)
            dis_loss.append(loss)

            if s % 1000 == 0:
                print('rec loss: {}, gen loss: {}, dis loss: {}'.\
                       format(np.mean(rec_loss), np.mean(gen_loss), np.mean(dis_loss)))

        print('rec loss: {}, gen loss: {}, dis loss: {}'.\
               format(np.mean(rec_loss), np.mean(gen_loss), np.mean(dis_loss)))
        rec.save_weights('weights_{}.h5'.format(i))

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

z = Input(shape=(m,))
h_decoded = decoder_h(z)
x_hat = decoder_mean(h_decoded)
decoder = Model(z, x_hat)
decoder.load_weights('weights_8.h5', by_name=True)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
fig.savefig('x.png')
