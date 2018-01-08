import tensorflow as tf
from speechrecproj.preprocessing.mfcc import tf_mfcc


def decoded_samples_preprocessing(decoded_samples, num_mel_bins=80, fft_resolution=256):
    """Applies mfcc and normalization."""
    tf.summary.audio(
        name="Input wav",
        tensor=tf.map_fn(lambda sample: tf.divide(sample, (2**16)/2), decoded_samples),
        sample_rate=16000
    )
    mfccs = tf_mfcc(
        decoded_samples,
        window_frame_length=200,
        stride=100,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600,
        fft_resolution=fft_resolution,
        num_mel_bins=num_mel_bins)
    # normalize

    normalized_mfcc = tf.nn.l2_normalize(mfccs, dim=[1, 2])
    tf.summary.image(
        name="Normalized MFCC",
        tensor=tf.expand_dims(tf.transpose(mfccs, perm=[0, 2, 1]), -1)
    )
    return normalized_mfcc
