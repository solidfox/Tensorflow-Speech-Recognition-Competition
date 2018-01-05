import tensorflow as tf


class TFrecord_reader():
    def __init__(self, filename):
        self.filename = filename

    def generate_datasets(self, valset_proportion):
        feature = {
            'label': tf.FixedLenFeature([], tf.int16),
            'wav': tf.FixedLenFeature([16000, ], tf.int16)
        }
        dataset = tf.data.TFRecordDataset([self.filename])
        dataset = dataset.map(lambda record: tf.parse_example(record, feature))
        # TODO Adding noise
        trainset = dataset.skip(valset_proportion)
        trainset = trainset.batch(32)
        valset = dataset.take(valset_proportion)
        valset = valset.batch(32)

        return trainset, valset