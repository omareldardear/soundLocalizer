import scipy.io.wavfile
from librosa.filters import mel

from utils import split_audio_chunks, filter_voice, get_power, butter_bandpass_filter, butter_lowpass_filter, butter_highpass_filter
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display
from gammatone.fftweight import fft_gtgram


def test_labels_from_azimuth(min_angles):
    azimuth_1 = -87
    azimuth_2 = 65

    class_1 = azimuth_1 + min_angles
    class_2 = azimuth_2 + min_angles

    # Testing with angle resolution 15
    assert (class_1 // 15) == 0
    assert (class_2 // 15) == 10

    # Testing with angle resolution 3
    assert (class_1 // 3) == 1
    assert (class_2 // 3) == 51

    print("Test labels PASSED !")



def test_VAD(background_filename, voice_filename):
    fs, signal_back = scipy.io.wavfile.read(background_filename, "wb", )
    fs, signal_voice = scipy.io.wavfile.read(voice_filename, "wb", )

    assert filter_voice(signal_back, fs) == False
    assert filter_voice(signal_voice, fs) == True

    sr, chunks_channel1, _ = split_audio_chunks(voice_filename, size_chunks=500, overlap=500)

    assert filter_voice(chunks_channel1[0], fs) == False
    assert filter_voice(chunks_channel1[1], fs) == False
    assert filter_voice(chunks_channel1[2], fs) == False
    assert filter_voice(chunks_channel1[3], fs) == True
    assert filter_voice(chunks_channel1[4], fs) == True
    assert filter_voice(chunks_channel1[5], fs) == True
    assert filter_voice(chunks_channel1[6], fs) == True
    assert filter_voice(chunks_channel1[7], fs) == True
    assert filter_voice(chunks_channel1[8], fs) == True
    assert filter_voice(chunks_channel1[9], fs) == True
    assert filter_voice(chunks_channel1[10], fs) == False


    print("Test VAD PASSED !!!")


def get_SNR():

    fs, signal = scipy.io.wavfile.read("/home/jonas/CLionProjects/soundLocalizer/dataset/background_test.wav", "wb", )
    fs, signal2 = scipy.io.wavfile.read("/home/jonas/CLionProjects/soundLocalizer/dataset/signal.wav", "wb", )

    power_noise = get_power(signal[:, 0])
    power_signal = get_power(signal2[:, 0])

    SNR = 10 * np.log10((power_signal - power_noise) / power_noise)

    print(f'SNR is {SNR}')


def main():
    # Testing generation of Labels from azimuth angles
    #test_labels_from_azimuth(90)

    # Testing google webRTC voice activity dector on background noise and voice noise
    #test_VAD("/home/jonas/CLionProjects/soundLocalizer/data/background_test.wav", "/home/jonas/CLionProjects/soundLocalizer/data/voice_test.wav")

    fs_back, signal_back = scipy.io.wavfile.read("/home/jonas/CLionProjects/soundLocalizer/dataset/background_test.wav", "wb", )
    signal_back = signal_back[:,1]
    signal_back = np.array(signal_back, dtype=np.float32)

    fs_v,  signal = scipy.io.wavfile.read("/home/jonas/CLionProjects/soundLocalizer/5.wav", "wb", )

    # Default gammatone-based spectrogram parameters
    twin = 0.08
    thop = twin / 2
    channels = 64
    fmin = 20

    gram  = fft_gtgram(signal, fs_v, twin, thop, channels, fmin)


    plt.show()

if __name__ == '__main__':
    main()
