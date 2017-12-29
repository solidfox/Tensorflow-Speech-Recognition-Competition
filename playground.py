from data import *
from data.labels import Label
from data.sample_files import SamplesManager


def main():
    sample_manager = SamplesManager('data')
    print(len(sample_manager.files_labels))
    print(Label.all_labels)


if __name__ == '__main__':
    main()