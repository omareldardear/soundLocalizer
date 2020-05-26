import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import split_audio_chunks, concat_fourier_transform
import numpy as np

plt.style.use('elegant.mplstyle')

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

def main(args):
    df = pd.read_csv(args.data_filename)
    
    show_fourrier_transform(df)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple visualization script ")
    parser.add_argument("data_filename", type=str,
                        help="Path filename of data")

    args = parser.parse_args()

    main(args)
