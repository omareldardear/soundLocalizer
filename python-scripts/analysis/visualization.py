import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
# Import for signal processing
import scipy.io.wavfile as wavfile
from scipy.signal import resample

import math
from utils import *
# plt.style.use('elegant.mplstyle')

def show_angles(df):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(x=df['label'], y=df['elevation'], marker='o', c=df['subject_id'], cmap=plt.get_cmap("inferno"))

    ax.legend(*sc.legend_elements())
    ax.set_title("Angles position per subject")
    ax.set_xlabel("Azimuth (in degree)")
    ax.set_ylabel("Elevation (in degree)")


def show_fourrier_transform(df):
    test_filename = df.iloc[50]['audio_filename']
    fs, sig1, sig2 = split_audio_chunks(test_filename)

    N = 512
    fft_sig1 = np.fft.fft(sig1[0], n=N)
    fft_sig2 = np.fft.fft(sig2[0], n=N)

    freqs = np.fft.fftfreq(N) * fs

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(3, 1, 1)
    sc = ax.stem(freqs[:int(len(fft_sig1)/2)], np.abs(fft_sig1[int(len(fft_sig1)/2):]))
    ax.set_title("Discrete Fourier Transform")
    ax.set_xlabel("Frequency in Hertz [Hz]")
    ax.set_ylabel("Amplitude")

    ax = fig.add_subplot(3, 1, 3)
    sc = ax.stem(freqs, np.abs(fft_sig2))
    ax.set_xlabel("Frequency in Hertz [Hz]")
    ax.set_ylabel("Amplitude")


def xcorr_freq(s1,s2):
    pad1 = np.zeros(len(s1))
    pad2 = np.zeros(len(s2))
    s1 = np.hstack([s1,pad1])
    s2 = np.hstack([pad2,s2])
    f_s1 =  np.fft.fft(s1)
    f_s2 =  np.fft.fft(s2)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = abs(f_s)
    denom[denom < 1e-6] = 1e-6
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    return np.abs( np.fft.ifft(f_s))[1:]


def test_gcc_phat(filename):
    test_filename = filename
    fs, signal = wavfile.read(test_filename)

    number_of_samples = round(len(signal) * float(8000) / fs)
    signal = np.array(resample(signal, number_of_samples), dtype=np.float32)

    max_tau = 0.0004
    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    signal1 = butter_bandpass_filter(signal1, 100, 900, fs)
    signal2 = butter_bandpass_filter(signal2, 100, 900, fs)

    refsig = np.linspace(1, 10, 10)

    delay, gcc = gcc_phat(signal1, signal2, fs, max_tau, n_delay=150)

    for d in delay:
        tmp = math.asin(d / max_tau)

        theta = tmp * 180 / math.pi
        print('\ntheta: {}'.format(int(theta)))

    limit = len(gcc) // 2
    x = (np.linspace(-limit, limit, num=len(gcc)) / fs) * 1000

    plt.plot(x, gcc)
    plt.show()


def main(args):


    filename = "/home/jonas/CLionProjects/soundLocalizer/data/dataset-500/68-4_41.wav"

    fs, signal = wavfile.read(filename)

    # number_of_samples = round(len(signal) * float(8000) / fs)
    # signal = np.array(resample(signal, number_of_samples), dtype=np.float32)

    max_tau = 0.0004
    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    gammatone_gcc, gammatone_delay = gcc_gammatoneFilter(signal1, signal2, fs, 40, 51)
    limit = len(gammatone_gcc[0]) // 2

    x = (np.linspace(-limit, limit, num=len(gammatone_gcc[0])) / fs)

    for gamma_d in gammatone_delay:
        for d in gamma_d:
            tmp = math.asin(d / max_tau)

            theta = tmp * 180 / math.pi
            print('\ntheta: {}'.format(int(theta)))

    for g in gammatone_gcc:
        plt.plot(x, g)

        plt.show()








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple visualization script ")
    parser.add_argument("data_filename", type=str,
                        help="Path filename of data")

    args = parser.parse_args()

    main(args)
