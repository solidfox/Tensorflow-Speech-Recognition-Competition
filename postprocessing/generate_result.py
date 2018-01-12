from glob import glob
import os
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf
import numpy as np
from speechrecproj.data.labels import Label
from os.path import dirname


def generate_result(estimator, test_data_dir, submission_filename='submission/submission.csv'):
    test_paths = glob(os.path.join(test_data_dir, '[!_]*/*wav'))

    if not os.path.exists(dirname(submission_filename)):
        raise AttributeError("The folder {} does not exists. Create it and run again.".format(dirname(submission_filename)))

    if os.path.exists(submission_filename):
        raise AttributeError("{} already exists. Delete it and run again.".format(submission_filename))

    fname, labels = [], []

    for path in test_paths:
        fname.append(os.path.basename(path))

    predictions = predict(estimator=estimator, test_paths=test_paths)

    for i, p in enumerate(predictions):
        print(i)
        labels.append(Label.from_index(p['label']).string)

    # Create the submission file
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = labels
    df.to_csv(submission_filename, index=False)


def test_input(test_paths):
    def intpu_fn():
        dataset = tf.contrib.data.Dataset.from_generator(
            generator=test_data_generator(test_paths),
            output_types=tf.float32,
            output_shapes=tf.TensorShape([16000])
        )
        return dataset.batch(128).make_one_shot_iterator().get_next()
    return intpu_fn


def test_data_generator(test_paths):
    def generator():
        for path in test_paths:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32)
            yield wav
    return generator


def predict(estimator, test_paths):
    return estimator.predict(
        input_fn=test_input(test_paths),
        predict_keys=None,
        hooks=None,
        checkpoint_path=None
    )
