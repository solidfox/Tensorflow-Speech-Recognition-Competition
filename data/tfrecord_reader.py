import tensorflow as tf


class TFRecordReader:
    def __init__(self, filename, validation_set_size, batch_size):
        self.validation_set_size = validation_set_size
        self.batch_size = batch_size
        _feature_schema = {
            'label': tf.FixedLenFeature([1], tf.int64),
            'wav': tf.FixedLenFeature([16000], tf.float32)
        }
        self.dataset = tf.data.TFRecordDataset([filename]) \
                         .map(lambda record: tf.parse_single_example(record, _feature_schema), num_parallel_calls=64)
        # for record in tf.python_io.tf_record_iterator(self.filename):
        #     print(tf.train.Example.FromString(record))
        #     break
        # TODO Adding noise

    def train_input_fn(self):
        with tf.name_scope('Training_data'):
            training_set = self.dataset.skip(self.validation_set_size) \
                                       .batch(self.batch_size)
            return training_set
            # iterator = training_set.make_initializable_iterator()



    def validation_input_fn(self):
        with tf.name_scope('Validation_data'):
            validation_set = self.dataset.take(self.validation_set_size) \
                                         .batch(self.batch_size)
            return validation_set

