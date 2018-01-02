__author__ = 'Daniel Schlaug'

import tensorflow as tf

from data.labels import Label


def convolutional_model_fn(preprocessed_voice_samples, labels, mode, dropout_rate, seed, learning_rate):
    """
    Generates a convolutional model.

    :param preprocessed_voice_samples: The inputs to the generated network.
    :param labels: The word-labels for the spoken samples.
    :param mode: A tf.estimator.ModeKeys value indicating if network is to be used for training, evaluation or prediction.
    :param dropout_rate: The fraction of nodes that are randomly deactivated at each iteration.
    :param seed: Seed for any random generators.
    :param learning_rate: The learning rate that the estimator will use.
    :return: The configured tensorflow network.
    """

    previous_layer = preprocessed_voice_samples

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
        activation=tf.nn.elu)

    previous_layer = tf.layers.dense(
        inputs=previous_layer,
        units=Label.n_labels(),
        activation=tf.nn.elu)

    logits = previous_layer

    return estimator_spec(labels, learning_rate, logits, mode)


def estimator_spec(labels, learning_rate, logits, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=tf.nn.softmax(logits, "softmax_predictions"))
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_operation = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_operation)

    else:  # mode == tf.estimator.ModeKeys.EVAL
        evaluation_metric_operation = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(inputs=logits, axis=1))}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=evaluation_metric_operation)

