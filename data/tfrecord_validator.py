import tensorflow as tf

from data import SamplesManager

__author__ = 'Daniel Schlaug'

if __name__ == '__main__':
    iterator = tf.python_io.tf_record_iterator("train.tfrecord")
    sm = SamplesManager("data")
    i = 0
    n_samples = 100
    for example_string in iterator:
        example = tf.train.Example.FromString(example_string)
        if i < n_samples:
            assert sm.files_labels[i].label.index == example.features.feature.get('label').int64_list.value[0]
            assert sm.files_labels[i].get_wav()[0] == example.features.feature.get('wav').float_list.value[0]
        else:
            break
        i = i + 1

    print "Validated the first {} samples".format(n_samples)