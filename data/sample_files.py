from glob import glob
from os.path import dirname, basename
import random
from labels import Label
import os
import tensorflow as tf
from tensorflow import data

from mfcc import create_mfcc
from resampling_ftt import resample

__author__ = 'Daniel Schlaug'


class Sample:
    def __init__(self, path):
        self.path = path
        self.label = Label.from_string(basename(dirname(path)))


class SamplesManager:
    """
    Responsible for making training data easy to access.
    """
    def __init__(self, data_dir):
        """
        :param data_dir: The parent folder of the training data. Should contain a folder called train with the training data.
        """
        all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

        self.files_labels = map(Sample, all_files)

        seed = 0
        rand = random.Random(seed)
        rand.shuffle(self.files_labels)
        valset_proportion = 0.1
        index = int(valset_proportion * len(self.files_labels))
        self.valset, self.trainset = self.files_labels[:index], self.files_labels[index:]

        def toDataset(samples):
            paths, labels, mfcc = [], [], []
            for sample in samples:
                paths.append(sample.path)
                labels.append(sample.label.index)
                # mfcc.append(create_mfcc(resample(sample[0], 8000), 8000, 128, 13))
            paths = tf.Variable(paths, dtype=tf.string)
            labels = tf.Variable(labels, dtype=tf.int32)
            return data.Dataset.from_tensors((paths, labels))

        self.valset = toDataset(self.valset)
        self.trainset = toDataset(self.trainset)
