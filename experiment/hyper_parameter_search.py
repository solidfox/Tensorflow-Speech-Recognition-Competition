import tensorflow as tf
from experiment_factory import ExperimentFactory
import environment
import data

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
            filename='data/train.tfrecord',
            validation_set_size=6000,
            batch_size=32
        )
        train_input_fn = dataset.next_training_batch
        eval_input_fn = dataset.next_validation_batch

        session_run_hook = CustomSessionRunHook()

        run_config = tf.contrib.learn.RunConfig(
            model_dir=env_conf.model_output_dir,
            save_summary_steps=100
            # save_checkpoints_steps=
        )

        # For Tensorboard
        summary_hook = tf.train.SummarySaverHook(
            save_steps=run_config.save_summary_steps,
            scaffold=tf.train.Scaffold(),
            summary_op=tf.summary.merge_all())

        experiment_factory = ExperimentFactory(
            environment_config=env_conf,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_hooks=[summary_hook],
            train_hooks=[session_run_hook]
        )

        # Create hyper parameter grids for each experiment.

        tf_h_params = tf.contrib.training.HParams(
            learning_rate=0.001,
            dropout_rate=0.3,
            random_seed=2222,
            training_steps=3000
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


if __name__ == '__main__':
    hyper_parameter_search()
