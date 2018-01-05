from glob import glob
import os
import pandas as pd

def generate_result(model, test_data_dir):
    # TODO: sub_path
    sub_path = "./submission/"
    test_paths = glob(os.path.join(test_data_dir, '*wav'))

    fname, results = [], []

    for path in test_paths:
        # TODO: Change depending on the model (path or wav), and fill the file
        predicts = model.predict(path)

        fname.extend(os.path.basename(path=path))
        results.extend(results)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = fname
    df['label'] = results
    df.to_csv(os.path.join(sub_path, 'submission.csv'), index=False)