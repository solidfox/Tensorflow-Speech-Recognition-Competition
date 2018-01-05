import random
import sys
from os.path import dirname, basename

import numpy as np
import tensorflow as tf
from scipy.io import wavfile


class TFrecord_Writer:
    def __init__(self, dataset, mode='train'):
        if mode == 'train':
            self.dataset = dataset
            self.filename = 'train.tfrecords'

    def write(self):

        writer = tf.python_io.TFRecordWriter(self.filename)

        for i in range(len(self.dataset)):
            if not i % 1000:
                print 'Train data: {}/{}'.format(i, len(self.dataset))
                sys.stdout.flush()

            if self.dataset[i].wav:
                wav = self.dataset[i].wav
            else:
                wav = loadwav(self.dataset[i].path)

            label = self.dataset[i].label.index

            feature = {
                'train/label': _int64_feature(label),
                'train/wav': _int64_feature_array(wav)
            }

            # example = tf.train.Example()
            # example.features.feature["label"].int64_list.value.append(label)
            # example.features.feature["wav"].int64_list.value.append(wav)
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()


def loadwav(path):
    sample_rate, samples = wavfile.read(path)
    if samples.shape[0] < sample_rate:
        for i in range(0, sample_rate - samples.shape[0]):
            samples = np.append(samples, 0)
    elif basename(dirname(path)) == '_background_noise_':
        index = random.randint(0, samples.shape[0] - 16000)
        samples = samples[index:index + 16000]
    return samples


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_array(array):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))
