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
    wavs, labels = tfreader.next_training_batch()

    experiment.hyper_parameter_search.hyper_parameter_search()

    # with tf.Session() as sess:
    #     result = sess.run(wavs)
    # print(result)
    # with tf.Session() as sess:
    #     result = sess.run(wavs)
    # print(result)

    # experiment.hyper_parameter_search.hyper_parameter_search(trainset, valset)

if __name__ == '__main__':
    main()
