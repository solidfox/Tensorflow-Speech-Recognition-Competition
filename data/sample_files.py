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

_seed = 0
_default_random_generator = random.Random(_seed)

class Sample:
    def __init__(self, path, random_generator=None):
        self.path = path
        self.label = Label.from_string(basename(dirname(path)))
        self.random = random_generator if random_generator is not None else _default_random_generator

    def __repr__(self):
        return "Sample({}, {})".format(self.path, self.label)

    def get_wav(self):
        if self.path == "":
            return np.zeros(16000)
        sample_rate, samples = wavfile.read(self.path)
        if samples.shape[0] < sample_rate:
            padding = np.zeros(sample_rate - samples.shape[0])
            np.concatenate((samples, padding), axis=0)
        elif basename(dirname(self.path)) == '_background_noise_':
            index = self.random.randint(0, samples.shape[0] - 16000)
            samples = samples[index:index + 16000]
        return samples



class SamplesManager:
    """
    Responsible for making training data easy to access.
    """

    def __init__(self, data_dir):
        """
        :param data_dir: The parent folder of the training data. Should contain a folder called train with the training data.
        """


        all_files = glob(os.path.join(data_dir, 'train/audio/[!_]*/*wav'))

        noises = glob(os.path.join(data_dir, 'train/audio/_*/*wav')) + [""]
        print(noises)
        desired_number_of_noise_samples = 2000
        repeats = desired_number_of_noise_samples / (len(noises))
        noises = np.repeat(noises, repeats)
        print("Noises data: " + str(len(noises)))

        self.files_labels = map(Sample, all_files)
        noises = map(Sample, noises)

        self.files_labels = np.concatenate((self.files_labels, noises))

        seed = 0
        rand = random.Random(seed)

        rand.shuffle(self.files_labels)


# TODO Move splitting logic into different file
#
#         index = int(valset_proportion * len(self.files_labels))
#         self.valset, self.trainset = self.files_labels[:index], self.files_labels[index:]
#
#         self.valset = samples_to_dataset(self.valset)
#         self.trainset = samples_to_dataset(self.trainset)
#
#
# def samples_to_dataset(samples):
#     labels, wavs = [], []
#     for sample in samples:
#         labels.append(sample.label.index)
#         wavs.append(sample.wav)
#     labels = tf.Variable(labels, dtype=tf.int16)
#     wavs = tf.convert_to_tensor(np.asarray(wavs))
#     return data.Dataset.from_tensor_slices((labels, wavs))