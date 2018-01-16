import tensorflow as tf

__author__ = 'Daniel Schlaug'


def estimator_spec(labels, learning_rate, logits, mode):
    export_outputs = {
        None: tf.estimator.export.ClassificationOutput(
            scores=logits,
            classes=None  # TODO
        )
    }

    with tf.name_scope('Prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=dict(
                    label=tf.argmax(logits, axis=-1)),
                export_outputs=export_outputs)

    with tf.name_scope('Loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        tf.summary.scalar('Loss', loss)

    with tf.name_scope('Optimization'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_operation = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            for v in tf.trainable_variables():
                print(v)
                variable_histogram(v)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_operation,
                export_outputs=export_outputs)

    with tf.name_scope('Evaluation'):
        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.argmax(input=logits, axis=1))
        evaluation_metric_operation = {
            'accuracy': accuracy}
        tf.summary.scalar('Accuracy', accuracy)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=evaluation_metric_operation,
            export_outputs=export_outputs)


def variable_histogram(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # /!\ Black magic ahead /!\
    tensor_name = var.name.replace(":0", "")
    print(tensor_name)
    tf.summary.histogram('histogram_' + tensor_name, var)
