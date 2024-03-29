import tensorflow as tf
from speechrecproj import preprocessing
from speechrecproj.data import Label
from speechrecproj.model.estimator_spec import estimator_spec

__author__ = 'Daniel Schlaug'


def convolutional_model_fn(features, labels, mode, params, config=None):
    """
    Configures a convolutional model.

    Args:
        features: A 2D tensor with the inputs to the generated network.
        labels: The word-labels for the spoken samples.
        mode: A tf.estimator.ModeKeys value indicating if network is to be used for training, evaluation or prediction.
        params (HParam): A tf.estimator.HParam object with the following hyperparameters:
            dropout_rate (float) The fraction of nodes that are randomly deactivated at each iteration.
            learning_rate (float) The learning rate that the estimator will use.
        config: Currently unused.

    Returns:
        An EstimatorSpec for the given input.
    """
    previous_layer = features['wav']

    previous_layer = preprocessing.decoded_samples_preprocessing(
        decoded_samples=previous_layer,
        num_mel_bins=80,
        fft_resolution=256

    )
    # Output shape: [None, 159, 13]

    with tf.name_scope('Convolutional_network'):
        previous_layer = tf.expand_dims(previous_layer, -1)

        previous_layer = tf.layers.conv2d(
            inputs=previous_layer,
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.elu)

        previous_layer = tf.layers.conv2d(
            inputs=previous_layer,
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.elu)

        previous_layer = tf.layers.max_pooling2d(
            inputs=previous_layer,
            pool_size=[2, 2],
            strides=[2, 2])

        previous_layer = tf.layers.conv2d(
            inputs=previous_layer,
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.elu)

        previous_layer = tf.layers.conv2d(
            inputs=previous_layer,
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.elu)

        previous_layer = tf.layers.max_pooling2d(
            inputs=previous_layer,
            pool_size=[2, 2],
            strides=[2, 2])

    with tf.name_scope('Fully_connected_network'):
        previous_layer = tf.layers.flatten(previous_layer)

        previous_layer = tf.layers.dense(
            inputs=previous_layer,
            units=200,
            activation=tf.nn.elu)

        previous_layer = tf.layers.dropout(
            inputs=previous_layer,
            rate=params.dropout_rate,
            training=mode == tf.estimator.ModeKeys.TRAIN)

        previous_layer = tf.layers.dense(
            inputs=previous_layer,
            units=200,
            activation=tf.nn.elu)

        previous_layer = tf.layers.dropout(
            inputs=previous_layer,
            rate=params.dropout_rate,
            training=mode == tf.estimator.ModeKeys.TRAIN)

        previous_layer = tf.layers.dense(
            inputs=previous_layer,
            units=Label.n_labels(),
            activation=tf.nn.elu)

        logits = previous_layer

    return estimator_spec(labels, params.learning_rate, logits, mode)
