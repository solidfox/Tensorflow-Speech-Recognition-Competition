from glob import glob
from os.path import dirname, basename
from random import shuffle
import labels as lbl
import os
import tensorflow as tf
from tensorflow import data

from mfcc import create_mfcc
from resampling_ftt import resample

__author__ = 'Daniel Schlaug'


class SamplesManager:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

        def label(sample_path):
            # Return the label for the specified path (based on the folder)
            return basename(dirname(sample_path))

        self.files_labels = map(
            lambda path: (path, lbl.Label.from_string(label(path)).index), all_files)

        shuffle(self.files_labels)
        valset_proportion = 0.1
        index = int(valset_proportion * len(self.files_labels))
        self.valset, self.trainset = self.files_labels[:index], self.files_labels[index:]

        def toDataset(samples):
            paths, labels, mfcc = [], [], []
            for sample in samples:
                paths.append(sample[0])
                labels.append(sample[1])
                # mfcc.append(create_mfcc(resample(sample[0], 8000), 8000, 128, 13))
            paths = tf.Variable(paths, dtype=tf.string)
            labels = tf.Variable(labels, dtype=tf.int32)
            return data.Dataset.from_tensors((paths, labels))

        self.valset = toDataset(self.valset)
        self.trainset = toDataset(self.trainset)
