from glob import glob
from os.path import dirname, basename
import labels as lbl
import os
import re

__author__ = 'Daniel Schlaug'


class SamplesManager:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

        def label(sample_path):
            # Return the label for the specified path (based on the folder)
            return basename(dirname(sample_path))

        self.files_labels = map(
            lambda path: (path, lbl.Label.from_string(label(path)).index, basename(path).split("_", 1)[0]), all_files)

        self.valset = []
        self.trainset = []

        with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
            validation_files = fin.readlines()

        valuid = set()

        for entry in validation_files:
            valuid.add(basename(entry).split("_", 1)[0])

        for sample in self.files_labels:
            if sample[2] in valuid:
                self.valset.append(sample[:2])
            else:
                self.trainset.append(sample[:2])
