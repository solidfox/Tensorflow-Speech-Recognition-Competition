from data import *

def main():
    sample_manager = SamplesManager('data')
    print(len(sample_manager.files_labels))
    print(sample_manager.files_labels[0])
    print(sample_manager.valset)
    print(sample_manager.trainset)
    print(Label.all_labels)

    # mfcc_test = create_mfcc(resample("data/train/audio/cat/300384f0_nohash_0.wav", 8000), 8000, 128, 13)
    # print(mfcc_test)
    # display_mfcc(mfcc_test)
    # show_fft_sample("data/train/audio/cat/300384f0_nohash_0.wav")


if __name__ == '__main__':
    main()