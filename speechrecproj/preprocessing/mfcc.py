import tensorflow as tf

def tf_mfcc(signals, window_frame_length, stride, lower_edge_hertz, upper_edge_hertz, fft_resolution, num_mel_bins, sample_rate=16000):
    """
    :param signals: [batch_size, number_samples]. Both batch_size and number_samples may be unknown.
    :param window_frame_length: frame_length
    :param stride: frame_step
    :param sample_rate: 16000
    :param lower_edge_hertz: 80.0 (first)
    :param upper_edge_hertz: 7600 (first)
    :param num_mel_bins: 80 (first)
    :return: mfccs
    """

    stfts = tf.contrib.signal.stft(
        signals,
        window_frame_length,
        stride,
        fft_length=fft_resolution
    )
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    # lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80  # TODO: Hyperparam?
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
    num_mfccs = 20
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., :num_mfccs]

    return mfccs