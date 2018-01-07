from glob import glob
import os
import pandas as pd
from scipy.io import wavfile
import tensorflow as tf
import numpy as np


def generate_result(estimator):
    sub_path = "submission/"
    test_data_dir = "data/test/"
    test_paths = glob(os.path.join(test_data_dir, '[!_]*/*wav'))

    fname, results, set = [], [], []

    # TODO: create wavs Tensors
    for path in test_paths:
        sample_rate, samples = wavfile.read(path)
        set.append(samples)

        # fname.extend(os.path.basename(path=path))
        # fname.extend(path)
        # results.extend(results)

    # predictions = estimator.predict(
    #     input_fn=test_set,
    #     predict_keys=None,
    #     hooks=None,
    #     checkpoint_path=None
    # )

    # Generator
    # for i in predictions:
    #     print(predictions)
    # Do the prediction


    # Create the submission file
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = results
    df.to_csv(os.path.join(sub_path, 'submission.csv'), index=False)


# if __name__ == '__main__':
#     generate_result()