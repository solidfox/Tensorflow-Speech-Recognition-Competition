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
    def __init__(self, path, wav=None):
        self.path = path
        self.label = Label.from_string(basename(dirname(path)))
        if wav is not None:
            self.wav = wav
        else:
            self.wav = loadwav(path)


class SamplesManager:
    """
    Responsible for making training data easy to access.
    """

    def __init__(self, data_dir, valset_proportion=0.1):
        """
        :param data_dir: The parent folder of the training data. Should contain a folder called train with the training data.
        """

        seed = 0
        rand = random.Random(seed)

        all_files = glob(os.path.join(data_dir, 'train/audio/[!_]*/*wav'))
        all_files = all_files[:10]

        noises = glob(os.path.join(data_dir, 'train/audio/_*/*wav'))
        noises = np.repeat(noises, 4)
        print("Noises data: " + str(len(noises)))

        self.files_labels = map(Sample, all_files)
        noises = map(Sample, noises)

        self.files_labels = np.concatenate((self.files_labels, noises))

        rand.shuffle(self.files_labels)
        index = int(valset_proportion * len(self.files_labels))
        self.valset, self.trainset = self.files_labels[:index], self.files_labels[index:]

        self.valset = samples_to_dataset(self.valset)
        self.trainset = samples_to_dataset(self.trainset)

        print(self.trainset)


def loadwav(path):
    sample_rate, samples = wavfile.read(path)
    if samples.shape[0] < sample_rate:
        for i in range(0, sample_rate - samples.shape[0]):
            samples = np.append(samples, 0)
    elif basename(dirname(path)) == '_background_noise_':
        index = random.randint(0, samples.shape[0] - 16000)
        samples = samples[index:index + 16000]
    return samples


def samples_to_dataset(samples):
    labels, wavs = [], []
    for sample in samples:
        labels.append(sample.label.index)
        wavs.append(sample.wav)
    labels = tf.Variable(labels, dtype=tf.int16)
    wavs = tf.convert_to_tensor(np.asarray(wavs))
    return data.Dataset.from_tensor_slices((labels, wavs))