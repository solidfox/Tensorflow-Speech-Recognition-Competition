import tensorflow as tf

__author__ = 'Daniel Schlaug'


class ExperimentFactory:
    def __init__(self, environment_config, train_input_fn, eval_input_fn,
                 eval_hooks, train_hooks, interleaved_eval_steps, interleaved_eval_frequency):
        self.environment_config = environment_config
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        self.eval_hooks = eval_hooks
        self.train_hooks = train_hooks
        self.interleaved_eval_steps = interleaved_eval_steps
        self.eval_frequency = interleaved_eval_frequency

    def experiment(self, estimator):
        """
        Args:
            estimator (tensorflow.estimator.Estimator):

        Returns:
            tensorflow.contrib.learn.Experiment: Experiment with the factory's presets and the given estimator.
        """
        return tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=self.train_input_fn,
            eval_input_fn=self.eval_input_fn,
            train_steps=estimator.params.training_steps,
            min_eval_frequency=self.eval_frequency,
            train_monitors=self.train_hooks,
            eval_hooks=self.eval_hooks,
            eval_steps=self.interleaved_eval_steps,
            checkpoint_and_export=False,
        )