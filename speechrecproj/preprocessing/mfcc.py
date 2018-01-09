import tensorflow as tf


def tf_mfcc(signals, window_frame_length=200, stride=100, lower_edge_hertz=80,
            upper_edge_hertz=7600, fft_resolution=256, num_mel_bins=80, sample_rate=16000):
    """
    :param signals: One batch of raw wav samples. Shape [?, 16000]
    :param window_frame_length: The number of samples to analyse for each Fourier Transform time step. Depends on the lowest pitch you want to be able to detect. For speech the lowest pitch is about 80Hz and at 16000Hz sampling that's a window of 200 samples.
    :param stride: The number of samples by which to slide the window after each fourier transform time step. Often about half the window_frame_length.
    :param sample_rate: The sample rate of the input signals.
    :param lower_edge_hertz: 80.0 (first)
    :param upper_edge_hertz: 7600 (first)
    :param num_mel_bins: 80 (first)
    :return: mfccs
    """

    # Fourier Transform
    stfts = tf.contrib.signal.stft(
        signals,
        window_frame_length,
        stride,
        fft_length=fft_resolution
    )
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value

    # lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    num_mfccs = 13
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., :num_mfccs]

    return mfccs