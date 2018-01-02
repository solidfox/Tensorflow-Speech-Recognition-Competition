# Math
import numpy as np
import librosa

# Visualization
import matplotlib.pyplot as plt
import librosa.display

# sample and sample_rate after the FTT
# n_mels=128 n_mfcc=13
def create_mfcc(sample, sample_rate, n_mels, n_mfcc):
    S = librosa.feature.melspectrogram(sample, sr=sample_rate, n_mels=n_mels)
    # Convert to log scale (dB). The reference chosen is the max peak power
    log_s = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_s, n_mfcc=n_mfcc)
    return mfcc

def display_mfcc(mfcc):
    # Let's pad on the first and second deltas while we're at it
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(delta2_mfcc)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()