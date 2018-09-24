"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     9/1/2018
Comments: Includes functions for defining the CNN layers
**********************************************************************************
"""
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten


def conv_layer_2d(x, num_filters, kernel_size, add_reg=False, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        regularizer = None
        if add_reg:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        net = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=kernel_size,
                               strides=stride, padding='SAME', kernel_regularizer=regularizer)
        print('{}: {}'.format(layer_name, net.get_shape()))
        return net


def conv_layer_3d(x, num_filters, kernel_size, add_reg=False, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        regularizer = None
        if add_reg:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        net = tf.layers.conv3d(inputs=x, filters=num_filters, kernel_size=kernel_size,
                               strides=stride, padding='SAME', kernel_regularizer=regularizer)
        print('{}: {}'.format(layer_name, net.get_shape()))
        return net


def fc_layer(x, num_units, add_reg, layer_name):
    with tf.name_scope(layer_name):
        regularizer = None
        if add_reg:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        net = tf.layers.dense(inputs=x, units=num_units, kernel_regularizer=regularizer)
        print('{}: {}'.format(layer_name, net.get_shape()))
        return net


def max_pool_2d(x, pool_size, stride, name, padding='VALID'):
    """Create a max pooling layer."""
    net = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride,
                                  padding=padding, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def max_pool_3d(x, pool_size, stride, name, padding='VALID'):
    """Create a max pooling layer."""
    net = tf.layers.max_pooling3d(inputs=x, pool_size=pool_size, strides=stride,
                                  padding=padding, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def average_pool_2d(x, pool_size, stride, name, padding='VALID'):
    """Create an average pooling layer."""
    net = tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride,
                                      padding=padding, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def average_pool_3d(x, pool_size, stride, name, padding='VALID'):
    """Create an average pooling layer."""
    net = tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=stride,
                                      padding=padding, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def global_average_pool(x, name='global_avg_pooling'):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)
    """
    net = global_avg_pool(x, name=name)
    print('{}: {}'.format(name, net.get_shape()))
    return net


def dropout(x, rate, training):
    """Create a dropout layer."""
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        out = tf.cond(training,
                      lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                      lambda: batch_norm(inputs=x, is_training=training, reuse=True))
        return out


def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)


def concatenation(layers):
    return tf.concat(layers, axis=3)


def relu(x):
    return tf.nn.relu(x)
