# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from prepare_dataset import maybe_download_and_extract

dataset = maybe_download_and_extract()
X = dataset['train_images']
Y = dataset['train_labels']
X_test = dataset['test_images']
Y_test = dataset['test_labels']

# Building 'AlexNet'
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 96, 3, strides=1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 20, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network,
                    checkpoint_path='checkpoints/model_alexnet',
                    max_checkpoints=1,
                    tensorboard_verbose=2,
                    tensorboard_dir='logs')

# model.load('model_save')
model.fit(X, Y,
          validation_set=(X_test, Y_test),
          batch_size=64,
          n_epoch=100,
          shuffle=True,
          show_metric=True,
          snapshot_step=500,
          snapshot_epoch=False,
          run_id='model_alexnet')

model.save('model_save')
