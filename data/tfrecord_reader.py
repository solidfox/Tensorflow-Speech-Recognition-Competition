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
        self.validation_set_size = validation_set_size
        self.batch_size = batch_size
        self.dataset = tf.data.TFRecordDataset([filename]) \
                         .map(_parse_tfrecord, num_parallel_calls=64)
        # for record in tf.python_io.tf_record_iterator(self.filename):
        #     print(tf.train.Example.FromString(record))
        #     break
        # TODO Adding noise

    def train_input_fn(self):
        with tf.name_scope('Training_data'):
            training_set = self.dataset.skip(self.validation_set_size) \
                                       .batch(self.batch_size)
            iterator = training_set.make_initializable_iterator()
            features, labels = iterator.get_next()
            return features, labels


    def validation_input_fn(self):
        with tf.name_scope('Validation_data'):
            validation_set = self.dataset.take(self.validation_set_size) \
                                         .batch(self.batch_size)
            iterator = validation_set.make_initializable_iterator()
            features, labels = iterator.get_next()
            return features, labels

