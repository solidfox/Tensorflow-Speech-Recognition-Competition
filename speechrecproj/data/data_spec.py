import tensorflow as tf
from speechrecproj.data.labels import Label

__author__ = 'Daniel Schlaug'


def labels_column():
    return tf.feature_column.categorical_column_with_vocabulary_list(
        key='label',
        vocabulary_list=map(lambda label: label.string, Label.all_labels),
        dtype=tf.int64)


def features_column():
    return tf.feature_column.numeric_column(
        key='wav',
        shape=16000,
        dtype=tf.float32)
