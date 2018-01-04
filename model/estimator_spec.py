import tensorflow as tf

__author__ = 'Daniel Schlaug'

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