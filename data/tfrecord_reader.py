import tensorflow as tf

def _parse_tfrecord(serialized_example):
    feature_schema = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'wav': tf.FixedLenFeature([16000], tf.float32)
    }
    example = tf.parse_single_example(serialized_example, feature_schema)
    return example["wav"], example["label"]


class TFRecordReader:
    def __init__(self, filename, validation_set_size, batch_size):
        self._filename = filename
        self._dataset = None
        self._training_set_iterator = None
        self._validation_set_iterator = None
        self.validation_set_size = validation_set_size
        self.batch_size = batch_size



    @property
    def dataset(self):
        with tf.name_scope('Whole_dataset'):
            if self._dataset is None or self._dataset.graph != tf.get_default_graph():
                self._dataset = tf.data.TFRecordDataset([self._filename]) \
                                  .map(_parse_tfrecord, num_parallel_calls=64)
            return self._dataset

    @property
    def training_set_iterator(self):
        with tf.name_scope('Training_data'):
            if self._training_set_iterator is None or self._training_set_iterator.graph != tf.get_default_graph():
                self._training_set_iterator = \
                    self.dataset.skip(self.validation_set_size) \
                                .batch(self.batch_size) \
                                .make_one_shot_iterator()
            return self._training_set_iterator

    @property
    def validation_set_iterator(self):
        with tf.name_scope('Validation_data'):
            if self._validation_set_iterator is None or self._validation_set_iterator.graph != tf.get_default_graph():
                self._validation_set_iterator = \
                    self.dataset.take(self.validation_set_size) \
                                .batch(self.batch_size) \
                                .make_one_shot_iterator()
            return self._validation_set_iterator

    def next_training_batch(self):
        return self.training_set_iterator.get_next()

    def next_validation_batch(self):
        return self.validation_set_iterator.get_next()

