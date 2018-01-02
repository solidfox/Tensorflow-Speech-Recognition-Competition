__author__ = 'Daniel Schlaug'

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPool2D, Dropout

from data.labels import Label

def convolutional_model_fn(input_layer, dropout_rate, seed, mode):
    """
    Generates a convolutional model.

    :param input_layer: The inputs to the generated network.
    :param dropout_rate: The fraction of nodes that are randomly deactivated at each iteration.
    :param seed: Seed for any random generators.
    :param mode: A tf.estimator.ModeKeys value indicating if network is to be used for training, evaluation or prediction.
    :return: The cofigured tensorflow network.
    """

    previous_layer = input_layer

    previous_layer = tf.layers.conv2d(
        inputs=previous_layer,
        filters=8,
        kernel_size=[8, 20],
        padding='same',
        activation=tf.nn.elu)

    previous_layer = tf.layers.max_pooling2d(
        inputs=previous_layer,
        pool_size=[8, 8],
        strides=[8, 8])

    previous_layer = tf.layers.conv2d(
        inputs=previous_layer,
        filters=64,
        kernel_size=[4, 4],
        padding='same',
        activation=tf.nn.elu)

    previous_layer = tf.layers.max_pooling2d(
        inputs=previous_layer,
        pool_size=[2, 2],
        strides=[2, 2])

    previous_layer = tf.layers.dense(
        inputs=previous_layer,
        units=200,
        activation=tf.nn.elu)

    previous_layer = tf.layers.dropout(
        inputs=previous_layer,
        rate=dropout_rate,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    previous_layer = tf.layers.dense(
        inputs=previous_layer,
        units=200,
        activation=tf.nn.softmax_cross_entropy_with_logits)

    previous_layer = tf.layers.dense(
        inputs=previous_layer,
        units=Label.n_labels(),
        activation=tf.nn.elu)

    logits = previous_layer

    if