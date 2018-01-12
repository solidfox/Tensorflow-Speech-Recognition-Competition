from speechrecproj.environment import running_on_hops

__author__ = 'Daniel Schlaug'


class EnvironmentConfig:
    def __init__(self):
        if running_on_hops:
            import hops
            self.output_dir = hops.tensorboard.logdir()
            self.input_dir = 'hdfs:///Projects/TF_Speech_Project/Tensorflow_Speech_Recognition'
        else:
            self.output_dir = 'output'
            self.input_dir = 'input'
