import sys
import os
from random import Random
from os.path import dirname, basename

import numpy as np
import scipy
import tensorflow as tf

from speechrecproj.data import SamplesManager


class TFRecordWriter:
    def __init__(self, dataset, output_file, random_gen = Random()):
        self.random_gen = random_gen
        self.dataset = dataset
        self.filename = output_file

    def write(self):

        if os.path.exists(self.filename):
            raise AttributeError("TFRecordWriter destination path {} already exists. Delete it and run again.".format(self.filename))

        writer = tf.python_io.TFRecordWriter(self.filename)

        n_wavs = len(self.dataset)

        for i in range(len(self.dataset)):
            if not i % 100:
                print "\r", 'Converting wavs: {}/{}'.format(i, n_wavs),

            wav = self.read_wav(self.dataset[i]).astype(np.float32)

            label = self.dataset[i].label.index

            feature = {
                'label': _int64_feature(label),
                'wav': _float_feature_array(wav)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()

    def read_wav(self, path):
        if path == "":
            return np.zeros(16000, dtype=np.int16)
        sample_rate, samples = scipy.io.wavfile.read(path)
        if samples.shape[0] < sample_rate:
            padding = np.zeros(sample_rate - samples.shape[0], dtype=np.int16)
            samples = np.concatenate((samples, padding), axis=0)
        elif basename(dirname(path)) == '_background_noise_':
            index = self.random_gen.randint(0, samples.shape[0] - 16000)
            samples = samples[index:index + 16000]
        assert len(samples) == 16000
        return samples


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature_array(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))


if __name__ == '__main__':
    sample_manager = SamplesManager("input")
    dataset = sample_manager.files_labels
    tf_writer = TFRecordWriter(dataset, "train.tfrecord")
    tf_writer.write()
