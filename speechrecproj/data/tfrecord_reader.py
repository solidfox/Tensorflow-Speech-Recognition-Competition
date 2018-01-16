import tensorflow as tf
from speechrecproj.data import data_spec


def _parse_example(serialized_example):
    feature_spec = tf.feature_column.make_parse_example_spec(
        feature_columns=data_spec.features_column() + data_spec.labels_column()
    )
    example = tf.parse_example(serialized_example, feature_spec)
    return dict(wav=example['wav']), example['label']


class TFRecordReader:
    def __init__(self, filename, validation_set_size, batch_size, epochs=None):
        self._filename = filename
        self._dataset = None
        self.validation_set_size = validation_set_size
        self.batch_size = batch_size
        self.epochs = epochs

    @property
    def dataset(self):
        with tf.name_scope('Whole_dataset'):
            if self._dataset is None:
                self._dataset = tf.data.TFRecordDataset(filenames=[self._filename], buffer_size=5*(10**8)) \
                                  .map(_parse_example, num_parallel_calls=64)
            return self._dataset

    def new_training_set_iterator(self):
        with tf.name_scope('Training_data'):
            return self.dataset.skip(self.validation_set_size) \
                               .repeat(self.epochs) \
                               .batch(self.batch_size) \
                               .make_one_shot_iterator()

    def new_validation_set_iterator(self):
        with tf.name_scope('Validation_data'):
            return self.dataset.take(self.validation_set_size) \
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

