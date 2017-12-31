from data import *


def main():
    sample_manager = SamplesManager('data')
    print(len(sample_manager.files_labels))
    print(sample_manager.files_labels[0])
    print(sample_manager.valset)
    print(sample_manager.trainset)
    print(Label.all_labels)


if __name__ == '__main__':
    main()