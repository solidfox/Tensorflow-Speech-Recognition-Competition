import tensorflow as tf

def _parse_tfrecord(serialized_example):
    feature_schema = {
        'label': tf.FixedLenFeature([1], tf.int64),
        'wav': tf.FixedLenFeature([16000], tf.float32)
    }
    example = tf.parse_single_example(serialized_example, feature_schema)
    return example['wav'], example['label']


class TFRecordReader:
    def __init__(self, filename, validation_set_size, batch_size):
        self._filename = filename
        self._dataset = None
        self._training_set_iterator = None
        self._validation_set_iterator = None
        self.validation_set_size = validation_set_size
        self.batch_size = batch_size

    def new_dataset(self):
        with tf.name_scope('Whole_dataset'):
            if self._dataset is None:
                self._dataset = tf.data.TFRecordDataset(filenames=[self._filename], buffer_size=5*(10**8)) \
                                  .map(_parse_tfrecord, num_parallel_calls=64)
            return self._dataset

    def new_training_set_iterator(self):
        with tf.name_scope('Training_data'):
            return self.new_dataset().skip(self.validation_set_size) \
                                     .repeat() \
                                     .batch(self.batch_size) \
                                     .make_one_shot_iterator()

    def new_validation_set_iterator(self):
        with tf.name_scope('Validation_data'):
            return self.new_dataset().take(self.validation_set_size) \
                                     .batch(self.batch_size) \
                                     .make_one_shot_iterator()

    def training_input_fn(self):
        iterator = self.new_training_set_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def validation_input_fn(self):
        iterator = self.new_validation_set_iterator()
        features, labels = iterator.get_next()
        return features, labels

