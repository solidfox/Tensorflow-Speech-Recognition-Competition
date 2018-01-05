import tensorflow as tf


class TFrecord_reader():
    def __init__(self, filename):
        self.filename = filename
        self.nbr_elements = 0
        for record in tf.python_io.tf_record_iterator(filename):
            self.nbr_elements += 1

    def generate_datasets(self, valset_proportion=0.1):
        feature = {
            'label': tf.FixedLenFeature([], tf.int16),
            'wav': tf.FixedLenFeature([16000, ], tf.int16)
        }
        dataset = tf.data.TFRecordDataset([self.filename])
        dataset = dataset.map(lambda record: tf.parse_example(record, feature))
        # TODO Adding noise
        index = int(self.nbr_elements * valset_proportion)
        trainset = dataset.skip(index)
        trainset = trainset.shuffle(1000)
        trainset = trainset.batch(32)
        valset = dataset.take(index)
        valset = valset.shuffle(1000)
        valset = valset.batch(32)

        return trainset, valset
