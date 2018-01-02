import numpy as np
from scipy.fftpack import fft

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def resample(file_path, new_sample_rate):
    sample_rate, samples = wavfile.read(file_path)
    p = int(float(new_sample_rate)/float(sample_rate) * samples.shape[0])
    return signal.resample(samples, p)

# y: sample, fs: sample rate
def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals


def show_fft_sample(file_path):
    # Resampling FTT
    new_sample_rate = 8000
    resampled = resample(file_path, new_sample_rate)

    xf, vals = custom_fft(resampled, new_sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title('FFT of recording sampled with ' + str(new_sample_rate) + ' Hz')
    plt.plot(xf, vals)
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()

