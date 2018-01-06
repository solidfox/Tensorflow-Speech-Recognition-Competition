from data import *
import experiment.hyper_parameter_search

signals = tf.placeholder(tf.float32, [None, 16000])


def main():

    sample_manager = SamplesManager('data')
    print(len(sample_manager.files_labels))
    print(sample_manager.files_labels[0])
    print(Label.all_labels)

    sample_manager.files_labels[0].get_wav()
    tfreader = TFRecordReader(filename='data/train.tfrecord', validation_set_size=6000, batch_size=600)
    first_batch = tfreader.validation_input_fn().make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        result = sess.run(first_batch)
    print(result)


    # experiment.hyper_parameter_search.hyper_parameter_search(trainset, valset)

    # mfccs = decoded_samples_preprocessing(signals)
    # tf_network = convolutional_model_fn(preprocessed_voice_samples=mfccs,
    #                                     labels=sample_manager.files_labels.map(lambda p, labels, w: labels), ...)

    # init = tf.initialize_all_variables()
    # sess = tf.InteractiveSession()
    # sess.run(init)

    # sess.run(tf_network, feed_dict={signals: sample_manager.files_labels.map(lambda p, l, wavs: wavs)})


if __name__ == '__main__':
    main()
