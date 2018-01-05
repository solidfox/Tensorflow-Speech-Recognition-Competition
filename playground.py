from data import *
import experiment.hyper_parameter_search

signals = tf.placeholder(tf.float32, [None, 16000])


def main():

    sample_manager = SamplesManager('data', 0.1)
    print(len(sample_manager.files_labels))
    print(sample_manager.files_labels[0])
    print(loadwav(sample_manager.files_labels[0].path).shape)
    print(sample_manager.valset)
    print(sample_manager.trainset)
    print(Label.all_labels)

    tfwriter = TFrecord_Writer(sample_manager.trainset, mode='train')
    # tfwriter.write()

    # experiment.hyper_parameter_search.hyper_parameter_search(sample_manager.trainset, sample_manager.valset)

    # mfccs = decoded_samples_preprocessing(signals)
    # tf_network = convolutional_model_fn(preprocessed_voice_samples=mfccs,
    #                                     labels=sample_manager.files_labels.map(lambda p, labels, w: labels), ...)

    # init = tf.initialize_all_variables()
    # sess = tf.InteractiveSession()
    # sess.run(init)

    # sess.run(tf_network, feed_dict={signals: sample_manager.files_labels.map(lambda p, l, wavs: wavs)})


if __name__ == '__main__':
    main()
