from glob import glob
import os
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn


def generate_result(estimator):
    sub_path = "submission/"
    test_data_dir = "data/test/"
    test_paths = glob(os.path.join(test_data_dir, '[!_]*/*wav'))

    fname, labels = [], []

    # for path in test_paths:
    #     sample_rate, samps = wavfile.read(path)
    #     samples.append(samps)
    #
    #     file_name = os.path.basename(path)
    #     print(file_name)
    #     fname.append(file_name)
    #
    # np_samples = np.array(samples)
    # print(np_samples)
    # print(np_samples.shape)

    # dictionary = create_dic(test_paths)
    # print(dictionary)

    predictions = predict(test_paths=test_paths, estimator=estimator)
    print(predictions)

    for p in predictions:
        print(p)


    # # Create the submission file
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = labels
    df.to_csv(os.path.join(sub_path, 'submission.csv'), index=False)


def create_dict(test_paths):
    def generator():
        for path in test_paths:
            _, sample = wavfile.read(path)
            sample = sample.astype(np.float32) / np.iinfo(np.int16).max
            fname = os.path.basename(path)
            yield dict(
                sample_name=np.string_(fname),
                sample=sample,
            )
    return generator


# def create_dic(test_paths):
#     return tf.estimator.inputs.numpy_input_fn(
#         x=test_data_generator(test_paths),
#         y=None,
#         shuffle=False,
#         num_epochs=1
#     )


def predict(test_paths, estimator):
    input_fn = generator_input_fn(
        x=create_dict(test_paths),
        batch_size=1,
        shuffle=False,
        num_epochs=1,
        queue_capacity=1,
        num_threads=1,
    )

    return estimator.predict(
        input_fn=input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None
    )
