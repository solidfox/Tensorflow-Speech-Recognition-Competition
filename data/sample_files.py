from glob import glob
from os.path import dirname, basename
import random
from labels import Label
import os
import tensorflow as tf
from tensorflow import data
import numpy as np
from scipy.io import wavfile

__author__ = 'Daniel Schlaug'


class Sample:
    def __init__(self, path):
        self.path = path
        self.label = Label.from_string(basename(dirname(path)))
        # self.wav = loadwav(path)


class SamplesManager:
    """
    Responsible for making training data easy to access.
    """

    def __init__(self, data_dir, valset_proportion=0.1):
        """
        :param data_dir: The parent folder of the training data. Should contain a folder called train with the training data.
        """

        all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

        self.files_labels = map(Sample, all_files)
        wavs = [f.path for f in self.files_labels]
        print(wavs[:3])

        count_larger(wavs)

        self.files_labels = self.files_labels[:10]

        seed = 0
        rand = random.Random(seed)
        rand.shuffle(self.files_labels)
        index = int(valset_proportion * len(self.files_labels))
        self.valset, self.trainset = self.files_labels[:index], self.files_labels[index:]

        def toDataset(samples):
            paths, labels, wavs = [], [], []
            for sample in samples:
                paths.append(sample.path)
                labels.append(sample.label.index)
                wavs.append(sample.wav)
            paths = tf.Variable(paths, dtype=tf.string)
            labels = tf.Variable(labels, dtype=tf.int32)
            wavs = tf.Variable(wavs, dtype=tf.float32)
            return data.Dataset.from_tensors((paths, labels, wavs))

        self.valset = toDataset(self.valset)
        self.trainset = toDataset(self.trainset)


def loadwav(path):
    sample_rate, samples = wavfile.read(path)
    if samples.shape[0] < sample_rate:
        for i in range(0, sample_rate - samples.shape[0]):
            samples = np.append(samples, 0)
    return samples


def add_noises(wav):
    return 0


def count_larger(wavs):
    num_of_larger = 0
    for wav in wavs:
        sample_rate, samples = wavfile.read(wav)
        if samples.shape[0] > sample_rate:
            num_of_larger += 1
    print('Number of recordings shorter than 1 second: ' + str(num_of_larger))
