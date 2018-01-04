import tensorflow as tf
import model

__author__ = 'Daniel Schlaug'


def convolutional_experiment(environment_config, conv_hyper_params, train_input_fn, eval_input_fn):
    estimator = tf.estimator.Estimator(
        model_fn=model.convolutional_model_fn,
        model_dir=environment_config.model_output_dir
    )

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=conv_hyper_params.training_steps,
        min_eval_frequency=environment_config.min_steps_between_evaluations,
        train_monitors=None,
        eval_hooks=None,
        eval_steps=None  # Use all evaluation samples
    )

    return experiment
