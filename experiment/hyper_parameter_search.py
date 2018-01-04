import tensorflow as tf
from experiment_factory import ExperimentFactory
import environment

__author__ = 'Daniel Schlaug'

def hyper_parameter_search():
    """
    Configures and runs the hyper-parameter search

    """
    # Set up the environment
    env_conf = environment.EnvironmentConfig()
    train_input_fn = 42  # TODO
    eval_input_fn = 42  # TODO

    run_config = tf.estimator.RunConfig(
        model_dir="model_output",
        save_summary_steps=100
        # save_checkpoints_steps=
    )

    experiment_factory = ExperimentFactory(
        environment_config=env_conf,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn
    )

    # Create hyper parameter grids for each experiment.

    tf_h_params = tf.contrib.training.HParams(
        learning_rate=0.001,
        dropout_rate =0.3,
        random_seed=2222
    )

    # Run experiments with all their corresponding parameter grids.
    # This is where TFLauncher comes in on hops

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_factory.convolutional_experiment,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=tf_h_params
    )

    # (Evaluate the results and spin off more experiments?)
