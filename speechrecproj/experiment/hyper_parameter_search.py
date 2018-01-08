import os
import tensorflow as tf

from experiment_factory import ExperimentFactory
from speechrecproj import environment, model, data

__author__ = 'Daniel Schlaug'

class CustomSessionRunHook(tf.train.SessionRunHook):
    def __init__(self):
        pass

    def after_create_session(self, session, coord):
        print "Session created"

def hyper_parameter_search():
    """
    Configures and runs the hyper-parameter search

    """
    g = tf.Graph()
    with g.as_default():
        # Set up the environment
        env_conf = environment.EnvironmentConfig()
        dataset = data.TFRecordReader(
            filename=os.path.join(env_conf.input_dir, 'train.tfrecord'),
            validation_set_size=6000,
            batch_size=128
        )

        tf.logging.set_verbosity(tf.logging.INFO)

        run_config = tf.contrib.learn.RunConfig(
            model_dir=env_conf.output_dir,
            save_summary_steps=10,
            # save_checkpoints_steps=
        )

        session_run_hook = CustomSessionRunHook()

        # For Tensorboard
        summary_hook = tf.train.SummarySaverHook(
            save_steps=run_config.save_summary_steps,
            scaffold=tf.train.Scaffold(),
            summary_op=tf.summary.merge_all())

        experiment_factory = ExperimentFactory(
            environment_config=env_conf,
            train_input_fn=dataset.training_input_fn,
            eval_input_fn=dataset.validation_input_fn,
            eval_hooks=[summary_hook],
            train_hooks=[session_run_hook],
            interleaved_eval_steps=1,
            interleaved_eval_frequency=50,
        )

        # Create hyper parameter grids for each experiment.

        hyper_params = tf.contrib.training.HParams(
            learning_rate=0.001,
            dropout_rate=0.3,
            random_seed=2222,
            training_steps=3000
        )

        # Run experiments with all their corresponding parameter grids.
        # This is where TFLauncher comes in on hops

        run_config = run_config.replace(tf_random_seed=hyper_params.random_seed)

        estimator = tf.estimator.Estimator(
            model_fn=model.convolutional_model_fn,
            params=hyper_params,
            config=run_config
        )

        experiment = experiment_factory.experiment(estimator)

        experiment.test()

        experiment.train_and_evaluate()

        print "Finished training"
        # (Evaluate the results and spin off more experiments?)


if __name__ == '__main__':
    hyper_parameter_search()
