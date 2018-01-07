from glob import glob
import os
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf
import numpy as np


def generate_result():
    sub_path = "submission/"
    test_data_dir = "data/test/"
    test_paths = glob(os.path.join(test_data_dir, '[!_]*/*wav'))

    fname, results = [], []

    names, wavs = decode_wav(test_paths)
    predictions = create_dic(wavs)

    for p in predictions:
        print(p)
        results.extend(p)
    for i in names:
        print(i)
        fname.extend(i)

    # Create the submission file
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = results
    df.to_csv(os.path.join(sub_path, 'submission.csv'), index=False)


def decode_wav(test_paths):
    names, wavs = [], []

    graph = tf.Graph()
    with graph.as_default():
        file_sample = tf.placeholder(dtype=tf.float32)
        file_name = tf.placeholder(dtype=tf.string)
        sample_rate, samples = wavfile.read(file_sample)
        wav = samples
        name = os.path.basename(file_name)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        for path in test_paths:
            names.append(session.run(name, feed_dict={file_name: path}))
            wavs.append(session.run(wav, feed_dict={file_sample: path}))
            # TODO: ending print

        session.close()

    return names, wavs


def create_dic(wavs):
    return tf.estimator.inputs.numpy_input_fn(
        x=wavs,
        y=None,
        shuffle=False,
        num_epochs=1
    )

if __name__ == '__main__':
    generate_result()