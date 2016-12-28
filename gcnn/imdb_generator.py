# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import gzip
import numpy as np
import random
from six.moves import cPickle


def set_unk(sent, vocab_size=None):
    return [w if not vocab_size or w <= vocab_size + 1 else 1 for w in sent]


def pad(xs, ys, max_len=None):
    new_xs = []
    for x in xs:
        if max_len:
            x = x[:max_len]
        new_xs.append(x)
    xs = new_xs
    max_len = max_len if max_len else max([len(x) for x in xs])

    xa = np.zeros((len(xs), max_len), dtype='int32')
    for i in range(len(xs)):
        for j, c in enumerate(xs[i]):
            xa[i, j] = c
    ya = np.asarray(ys, dtype='int32') if ys else None

    return xa, ya


def add_eos(xs, max_len=None, eos=None):
    new_xs, new_ys = []
    for x in xs:
        y = x + [eos]
        x = [eos] + x
        new_xs.append(x)
        new_ys.apped(y)
    xs, ys = new_xs, new_ys
    max_len = max_len + 1 if max_len else max(max([len(x) for x in xs]))

    xa = np.zeros((len(xs), max_len), dtype='int32')
    ya = np.zeros((len(ys), max_len), dtype='int32')
    for i in range(len(xa)):
        for j, c in enumerate(xs[i]):
            xa[i, j] = c
            ya[i, j] = ys[i][j]

    return xa, ya


class IMDBLM(object):
    '''IMDB for training language model
    '''
    def __init__(self, path, which_set=None, train_ratio=0.9, 
                 max_len=5, vocab_size=101745, batch_size=10,
                 shuffle=True, seed=1111):
        self.__dict__.update(locals())
        self.__dict__.pop('self')
        
        self.random = random.Random()
        self.random.seed(self.seed)

        self.path = os.path.expandvars(self.path)
        
    def __call__(self):
        if self.path.endswith('.gz'):
            with gzip.open(self.path, 'r') as fp:
                data = cPickle.load(fp)
        else:
            with open(self.path, 'r') as fp:
                data = cPickle.load(fp)
            
        assert self.which_set in ['train', 'validation', 'test', None], \
                "which_set should be 'train' or 'validation' or 'test', " + \
                "but '{}' is given.".format(self.which_set)

        if self.which_set in ['train', 'validation', None]:
            train_x, train_x_unsup = data['train_x'], data['train_x_unsup']
            # concatenate all sentences
            text = []
            for x in train_x + train_x_unsup:
                text += x
            
            NUM_TRAIN = int(len(text) * self.train_ratio)
            NUM_VALID = len(text) - NUM_TRAIN
            if self.which_set == 'train':
                text = text[:NUM_TRAIN]
            elif self.which_set ==  'validation':
                text = text[-NUM_VALID:]
        else:
            # concatenate all sentences
            text = []
            for x in data['test_x']:
                text += x

        text = set_unk(text, self.vocab_size)

        del data
    
        nb_words = len(text)
        seg_size = nb_words // self.batch_size
        cursors = [i * seg_size  for i in range(self.batch_size)]

        while True:
            x = np.zeros((self.batch_size, self.max_len), dtype='int32')
            y = np.zeros((self.batch_size, self.max_len, 1), dtype='int32')
            for i in range(self.batch_size):
                c = cursors[i]
                for j in range(self.max_len):
                    x[i, j] = text[(c + j) % nb_words]
                    y[i, j, 0] = text[(c + j + 1) % nb_words]
                cursors[i] = (c + self.max_len) % nb_words
            yield x, y
