from glob import glob
import os
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf
import numpy as np
from speechrecproj.data.labels import Label

sub_path = "submission/"
test_data_dir = "data/test/"
test_paths = glob(os.path.join(test_data_dir, '[!_]*/*wav'))
fname, labels, samps = [], [], []


def generate_result(estimator):
    for path in test_paths:
        _, sample = wavfile.read(path)
        fname.append(os.path.basename(path))
        # samps.append(sample)

    predictions = predict(estimator=estimator)

    for p in predictions:
        print(Label.from_index(p['label']).string)
        labels.append(Label.from_index(p['label']).string)


    # Create the submission file
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = labels
    df.to_csv(os.path.join(sub_path, 'submission.csv'), index=False)

def test_input():
    # samples = tf.constant(np.array(samps), dtype=tf.float32)
    # dataset = tf.contrib.data.Dataset.from_tensor_slices((samples,))
    dataset = tf.contrib.data.Dataset.from_generator(
        generator=test_data_generator(),
        output_types=tf.float32,
        output_shapes=tf.TensorShape([16000])
    )
    return dataset.batch(1).make_one_shot_iterator().get_next()

def test_data_generator():
    def generator():
        for path in test_paths:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32)
            yield wav
    return generator

def predict(estimator):
    return estimator.predict(
        input_fn=test_input,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None
    )
