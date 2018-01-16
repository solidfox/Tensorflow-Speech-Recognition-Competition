import os
from os.path import dirname
from glob import glob

import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tensorflow.contrib.predictor import from_saved_model

from speechrecproj.environment.config import EnvironmentConfig
from speechrecproj.data.labels import Label


def generate_result(test_data_dir, submission_filename='submission/submission.csv'):
    env = EnvironmentConfig()

    submission_filename = os.path.join(env.output_dir, 'submission.csv')
    if not os.path.exists(dirname(submission_filename)):
        raise AttributeError("The folder {} does not exists. Create it and run again.".format(dirname(submission_filename)))
    if os.path.exists(submission_filename):
        raise AttributeError("{} already exists. Delete it and run again.".format(submission_filename))

    test_paths = glob(os.path.join(test_data_dir, '[!_]*/*wav'))
    generator = test_data_generator(test_paths)

    predictor = from_saved_model(
        export_dir=os.path.join(env.output_dir, '1516090613')
    )

    labels = []

    for i, wavs in enumerate(generator()):
        if i % 100 == 0:
            print(i)
        dictionary = dict(
            wav=wavs
        )
        prediction = predictor(dictionary)
        labels.append(Label.from_index(prediction['label']).string)

    # Create the submission file
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = map(os.path.basename, test_paths)
    df['label'] = labels
    df.to_csv(submission_filename, index=False)


# def test_input(test_paths):
#     def input_fn():
#         dataset = tf.contrib.data.Dataset.from_generator(
#             generator=test_data_generator(test_paths),
#             output_types=tf.float32,
#             output_shapes=tf.TensorShape([16000])
#         )
#         return dataset.batch(128).make_one_shot_iterator().get_next()
#     return input_fn


def test_data_generator(test_paths):
    def generator():
        for path in test_paths:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32)
            yield wav
    return generator

#
# def predict(estimator, test_paths):
#     return estimator.predict(
#         input_fn=test_input(test_paths),
#         predict_keys=None,
#         hooks=None,
#         checkpoint_path=None
#     )

if __name__ == '__main__':
    env = EnvironmentConfig()
    generate_result(test_data_dir=os.path.join(env.input_dir, "test"))
