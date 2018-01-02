__author__ = 'Daniel Schlaug'

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPool2D, Dropout

from data.labels import Label

def convolutional_model(input_layer, seed):
    model = Sequential()

    model.add(
        Conv2D(
            filters=8,
            kernel_size=(8, 20),
            activation='relu'))

    model.add(
        MaxPool2D(
            pool_size=(8, 8),
            strides=(8, 8)))

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(4, 4),
            activation='relu'))

    model.add(
        MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)))

    model.add(
        Dense(
            units=200,
            activation='relu'))

    model.add(
        Dropout(
            rate=0.30,
            seed=seed))

    model.add(
        tf.keras.layers.S)

    model.compile(
        optimizer=tf.keras.optimizers.Adam
    )


