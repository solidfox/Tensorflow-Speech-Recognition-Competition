import tensorflow as tf
from preprocessing.mfcc import tf_mfcc


def preprocessing_graph(filepathdataset):
    filepathdataset.map(read_wav, num_parallel_calls=16)

def read_wav(filepath, label):
    pcm_data = tf.read_file(filepath)
    audio_decoded = tf.contrib_audio.decode_wav(pcm_data)
    return audio_decoded, label

def wav_to_mfcc():
    signals = tf.placeholder(tf.float32, [None, 16000])
    tf_mfcc(signals, window_frame_length=200, stride=100, lower_edge_hertz=80.0, upper_edge_hertz=7600, num_mel_bins=80)