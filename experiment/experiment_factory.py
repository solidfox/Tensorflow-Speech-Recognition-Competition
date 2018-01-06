import tensorflow as tf
import model

__author__ = 'Daniel Schlaug'


class ExperimentFactory:
    def __init__(self, environment_config, train_input_fn, eval_input_fn):
        self.environment_config = environment_config
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn

    def convolutional_experiment(self, run_config, hyper_params):
        """Create an experiment to train and evaluate the model.

        Args:
            run_config (RunConfig): Configuration for Estimator run.
            hyper_params (HParam): Hyperparameters

        Returns:
            (Experiment) Experiment for training the mnist model.
        """

        run_config = run_config.replace(tf_random_seed=hyper_params.random_seed)

        estimator = tf.estimator.Estimator(
            model_fn=model.convolutional_model_fn,
            params=hyper_params,
            config=run_config
        )

        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=self.train_input_fn,
            eval_input_fn=self.eval_input_fn,
            train_steps=hyper_params.training_steps,
            min_eval_frequency=run_config.save_summary_steps,
            train_monitors=None,
            eval_hooks=None,
            eval_steps=None  # Use all evaluation samples
        )

        return experiment
