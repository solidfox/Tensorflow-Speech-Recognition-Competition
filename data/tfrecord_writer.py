import sys
import os

import numpy as np
import tensorflow as tf

from data import SamplesManager


class TFrecord_Writer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.filename = 'train.tfrecords'

    def write(self):

        if os.path.exists(self.filename):
            os.remove(self.filename)
        writer = tf.python_io.TFRecordWriter(self.filename)

        n_wavs = len(self.dataset)

        for i in range(len(self.dataset)):
            if not i % 100:
                print "\r", 'Converting wavs: {}/{}'.format(i, n_wavs),

            wav = self.dataset[i].get_wav()
            wav = wav.astype(np.float32)

            label = self.dataset[i].label.index

            feature = {
                'label': _int64_feature(label),
                'wav': _float_feature_array(wav)
            }

            # example = tf.train.Example()
            # example.features.feature["label"].int64_list.value.append(label)
            # example.features.feature["wav"].int64_list.value.append(wav)
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature_array(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))

if __name__ == '__main__':
    sample_manager = SamplesManager("data")
    dataset = sample_manager.files_labels
    tfwriter = TFrecord_Writer(dataset)
    tfwriter.write()