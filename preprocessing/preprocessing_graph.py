import tensorflow as tf
from preprocessing.mfcc import tf_mfcc


def decoded_samples_preprocessing(decoded_samples):
    mfccs = tf_mfcc(
        decoded_samples,
        window_frame_length=200,
        stride=100,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600,
        fft_resolution=256,
        num_mel_bins=80)

    # normalize
    # TODO: dimension
    normalized_mfcc = tf.nn.l2_normalize(mfccs, dim=0)
    return normalized_mfcc