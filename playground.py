from data import *


def main():
    sample_manager = SamplesManager('data')
    print(len(sample_manager.files_labels))
    print(Label.all_labels)


if __name__ == '__main__':
    main()