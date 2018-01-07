import environment

__author__ = 'Daniel Schlaug'


class EnvironmentConfig:
    def __init__(self):
        if environment.running_on_hops:
            import hops
            self.model_output_dir = hops.tensorboard.logdir()
        else:
            self.model_output_dir = 'model_output'