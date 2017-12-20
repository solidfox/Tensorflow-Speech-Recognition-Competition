from glob import glob
from os.path import dirname, basename
import os
import re

__author__ = 'Daniel Schlaug'


class SamplesManager:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
        def label(sample_path):
            return basename(dirname(sample_path))
        self.files_labels = map(lambda path: (path, label(path)), all_files)
