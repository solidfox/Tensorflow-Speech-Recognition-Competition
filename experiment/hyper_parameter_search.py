import tensorflow as tf
import experiment_factory
import environment

__author__ = 'Daniel Schlaug'

def hyper_parameter_search():
    """
    Configures and runs the hyper-parameter search

    """
    env_conf = environment.EnvironmentConfig()

    run_config = tf.estimator.RunConfig(
        model_dir="model_output",
        tf_random_seed=2222,
        save_summary_steps=100
        # save_checkpoints_steps=
    )

    tf_h_params = tf.contrib.training.HParams(
        learning_rate
    )

    # 1. Create different experiments.
    convolutional_experiment = experiment_factory.convolutional_experiment(
        conv_hyper_params=,
        environment_config=)

    # 2. Create hyper parameter grid for each experiment.

    # 3. Run each experiment with all the parameters in its grid.

    # (4. Evaluate the results and spin off more experiments?)




def run_experiment(experiment, hparams):
