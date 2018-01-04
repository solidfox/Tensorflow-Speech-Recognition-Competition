import tensorflow as tf
from data.labels import Label
from model.estimator_spec import estimator_spec

__author__ = 'Daniel Schlaug'


def convolutional_model_fn(preprocessed_voice_samples, labels, mode, params, config=None):
    """
    Configures a convolutional model.

    Args:
        preprocessed_voice_samples: A 2D tensor with the inputs to the generated network.
        labels: The word-labels for the spoken samples.
        mode: A tf.estimator.ModeKeys value indicating if network is to be used for training, evaluation or prediction.
        params: A dict of hyperparameters on the following form:
            dict(
                dropout_rate=(Float) The fraction of nodes that are randomly deactivated at each iteration.
                seed=(Int) Seed for any random generators.
                learning_rate=(Float) The learning rate that the estimator will use.
            )
        config: Currently unused.

    Returns:
        An EstimatorSpec for the given input.
    """
    tf.random_seed(params["seed"])

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
        rate=params["dropout_rate"],
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

    return estimator_spec(labels, params["learning_rate"], logits, mode)
