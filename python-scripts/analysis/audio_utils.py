import scipy.io.wavfile as wavfile
import webrtcvad
from scipy.signal import filtfilt, butter, lfilter

import collections
import contextlib
import wave
import time
import numpy as np

from datetime import timedelta as td
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import scipy.signal
import apkit


import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from analysis.CONFIG import *
from analysis.gammatone.fftweight import fft_gtgram
from analysis.gammatone.gtgram import gtgram
from scipy.signal import fftconvolve



#########################################################################################
# AUDIO PROCESSING FUNCTIONS                                                            #
#########################################################################################


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    # print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal


def remove_noise_stereo_signal(stereo_signal, noise_signal):
    data = stereo_signal[:, 0]
    data = data / 32768

    data2 = stereo_signal[:, 1]
    data2 = data2 / 32768


    output_1 = removeNoise(audio_clip=data, noise_clip=noise_signal)
    output_2 = removeNoise(audio_clip=data2, noise_clip=noise_signal)

    output = np.stack((output_1, output_2), axis=-1)


    return output

def normalize_audio(audio):
    first_channel = audio[:, 0]
    second_channel = audio[:, 1]

    first_audio = first_channel / np.max(np.abs(first_channel))
    second_audio = second_channel / np.max(np.abs(second_channel))

    norm_audio = np.stack((first_audio, second_audio), axis=1)

    return norm_audio


# noinspection PyTupleAssignmentBalance
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# noinspection PyTupleAssignmentBalance
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_power(signal):
    power = 0
    for i in signal:
        power += i ** 2

    return power / len(signal)


def FilteredSignal(signal, fs, cutoff):
    B, A = butter(1, cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(B, A, signal, axis=0)
    return filtered_signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def split_audio_chunks(audio_filename, stop_length, size_chunks=500, overlap=500):
    fs, signal = wavfile.read(audio_filename)

    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    length_chunk = int((fs * size_chunks) / 1000)
    overlap = int((fs * overlap) / 1000)

    index_start = 0
    chunk_signal1 = []
    chunk_signal2 = []

    if stop_length:
        stop_length = stop_length * fs
    else:
        stop_length = len(signal)
    while (index_start + length_chunk) < stop_length:

        stop_index = index_start + length_chunk
        chunck_s1 = signal1[index_start:stop_index]
        chunck_s2 = signal2[index_start:stop_index]

        if len(chunck_s1) == len(chunck_s2) == length_chunk:
            chunk_signal1.append(chunck_s1)
            chunk_signal2.append(chunck_s2)
        else:
            break
        index_start += overlap



    return fs, chunk_signal1, chunk_signal2


def pitch_augment(data, sampling_rate, pitch_factor):
    signal1 = librosa.effects.pitch_shift(data[:, 0].astype(float), sampling_rate, pitch_factor)
    signal2 = librosa.effects.pitch_shift(data[:, 1].astype(float), sampling_rate, pitch_factor)

    data = np.stack((signal1, signal2), axis=1)

    return data


def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

#########################################################################################
# FEATURES EXTRACTIONS FUNCTIONS                                                        #
#########################################################################################

def get_fft_gram(signal,  fs, time_window=0.08, channels=1024, freq_min=20):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    gammatone subband filters.
    """

    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    thop = time_window / 2

    fft_gram1 = fft_gtgram(signal1, fs, time_window, thop, channels, freq_min)
    fft_gram2 = fft_gtgram(signal2, fs, time_window, thop, channels, freq_min)

    fft_gram1 = np.flipud(20 * np.log10(fft_gram1))
    fft_gram2 = np.flipud(20 * np.log10(fft_gram2))

    return fft_gram1, fft_gram2


def get_MFCC(sample, sample_rate=16000, nb_mfcc_features=52):
    """
    Use librosa to compute MFCC features from an audio array with sample rate and number_mfcc
    :param sample:
    :param sample_rate:
    :param nb_mfcc_features:
    :return: np array of MFCC features
    """
    mfcc_feat = librosa.feature.mfcc(sample, sr=sample_rate, n_mfcc=nb_mfcc_features)
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat


def normalize_frames(signal, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])


def getArrayTDOA(signal, fs, windows_length=50, window_hop=25):
    signal1 = signal[:, 0]
    signal2 = signal[:, 1]


    length_chunk = int((fs * windows_length) / 1000)
    overlap = int((fs * window_hop) / 1000)

    index_start = 0
    tau_values = []

    while (index_start + length_chunk) < len(signal1):
        stop_index = index_start + length_chunk
        tmp_signal1 = signal1[index_start:stop_index]
        tmp_signal2 = signal2[index_start:stop_index]
        tau = tdoa(np.array(tmp_signal1), np.array(tmp_signal2), fs=fs)
        tau_values.append(tau)

        index_start += overlap

    return tau_values


def tdoa(x1, x2, interp=1, fs=1, phat=True):
    '''
    This function computes the time difference of arrival (TDOA)
    of the signal at the two microphones. This in turns is used to infer
    the direction of arrival (DOA) of the signal.

    Specifically if s(k) is the signal at the reference microphone and
    s_2(k) at the second microphone, then for signal arriving with DOA
    theta we have

    s_2(k) = s(k - tau)

    with

    tau = fs*d*sin(theta)/c

    where d is the distance between the two microphones and c the speed of sound.

    We recover tau using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)
    method. The reference is

    Knapp, C., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay.

    Parameters
    ----------
    x1 : nd-array
        The signal of the reference microphone
    x2 : nd-array
        The signal of the second microphone
    interp : int, optional (default 1)
        The interpolation value for the cross-correlation, it can
        improve the time resolution (and hence DOA resolution)
    fs : int, optional (default 44100 Hz)
        The sampling frequency of the input signal

    Return
    ------
    theta : float
        the angle of arrival (in radian (I think))
    pwr : float
        the magnitude of the maximum cross correlation coefficient
    delay : float
        the delay between the two microphones (in seconds)
    '''

    # zero padded length for the FFT
    n = (x1.shape[0] + x2.shape[0] - 1)
    if n % 2 != 0:
        n += 1

    # Generalized Cross Correlation Phase Transform
    # Used to find the delay between the two microphones
    # up to line 71
    X1 = np.fft.rfft(np.array(x1, dtype=np.float32), n=n)
    X2 = np.fft.rfft(np.array(x2, dtype=np.float32), n=n)

    if phat:
        X1 /= np.abs(X1)
        X2 /= np.abs(X2)

    cc = np.fft.irfft(X1 * np.conj(X2), n=interp * n)

    # maximum possible delay given distance between microphones
    t_max = n // 2 + 1

    # reorder the cross-correlation coefficients
    cc = np.concatenate((cc[-t_max:], cc[:t_max]))

    # pick max cross correlation index as delay
    tau = np.argmax(np.abs(cc))
    pwr = np.abs(cc[tau])
    tau -= t_max  # because zero time is at the center of the array

    return tau / (fs*interp)


def get_fbanks_gcc(signal, fs, win_size=1024, hop_size=512, nfbank=50, zoom=25, eps=1e-8):
    _FREQ_MAX = 8000
    _FREQ_MIN = 100

    tf = apkit.stft(signal, apkit.cola_hamming, win_size, hop_size)
    nch, nframe, _ = tf.shape

    # trim freq bins
    nfbin = int(_FREQ_MAX * win_size / fs)  # 0-8kHz
    freq = np.fft.fftfreq(win_size)
    freq = freq[:nfbin]
    tf = tf[:, :, :nfbin]

    # compute pairwise gcc on f-banks
    ecov = apkit.empirical_cov_mat(tf, fw=1, tw=1)
    fbw = apkit.mel_freq_fbank_weight(nfbank, freq, fs, fmax=_FREQ_MAX,
                                      fmin=_FREQ_MIN)
    fbcc = apkit.gcc_phat_fbanks(ecov, fbw, zoom, freq, eps=eps)

    # merge to a single numpy array, indexed by 'tpbd'
    #                                           (time, pair, bank, delay)
    feature = np.asarray([fbcc[(i, j)] for i in range(nch)
                          for j in range(nch)
                          if i < j])

    feature = np.squeeze(feature,axis=0)
    feature = np.moveaxis(feature, 2, 0)

    # and map [-1.0, 1.0] to 16-bit integer, to save storage space
    dtype = np.int16
    vmax = np.iinfo(dtype).max
    feature = (feature * vmax).astype(dtype)

    return feature

def gcc_phat(sig, refsig, fs=1, max_tau=0.00040, interp=16, n_delay=18):
    """
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    """

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(n // 2 + 1)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))

    ind = np.argpartition(cc, -n_delay)[-n_delay:]

    delay = []
    for i in ind:
        # find max cross correlation index
        shift = i - max_shift

        tau = shift / float(interp * fs)
        delay.append(tau)

    # find max cross correlation index
    shift = np.argmax(cc) - max_shift

    tau = shift / float(interp * fs)

    return delay, cc


def concat_fourier_transform(sig1, sig2, n=512):
    # Generalized Cross Correlation Phase Transform
    fft_sig1 = np.fft.fft(sig1, n=n)
    fft_sig2 = np.fft.fft(sig2, n=n)

    fft_phase = np.angle(fft_sig1)[int(len(fft_sig1) / 2):]
    fft_phase2 = np.angle(fft_sig2)[int(len(fft_sig2) / 2):]

    stack_fft = np.vstack((np.array(fft_phase), np.array(fft_phase2)))

    return stack_fft


def gcc_gammatoneFilter(sig1, sig2, fs, nb_bands, ndelay):
    gamma_sig1 = ToolGammatoneFb(sig1, fs, iNumBands=nb_bands)
    gamma_sig2 = ToolGammatoneFb(sig2, fs, iNumBands=nb_bands)

    gcc_features = []
    delay_features = []
    for filter1, filter2 in zip(gamma_sig1, gamma_sig2):
        delay, gcc = gcc_phat(filter1, filter2, fs, n_delay=ndelay)
        gcc = gcc * 10
        gcc_features.append(gcc)
        delay_features.append(delay)

    gcc_features = np.array(gcc_features)
    delay_features = np.array(delay_features)
    return gcc_features, delay_features


# see function mfcc.m from Slaneys Auditory Toolbox (Matlab)
def ToolGammatoneFb(afAudioData, f_s, iNumBands=26, f_low=1500):
    # initialization
    fEarQ = 9.26449
    fBW = 24.7
    iOrder = 1
    T = 1 / f_s

    # allocate output memory
    X = np.zeros([iNumBands, afAudioData.shape[0]])

    # compute the mid frequencies
    f_c = getMidFrequencies(f_low, f_s / 2, iNumBands, fEarQ, fBW)

    # compute the coefficients
    [afCoeffB, afCoeffA] = getCoeffs(f_c,
                                     1.019 * 2 * np.pi * (((f_c / fEarQ) ** iOrder + fBW ** iOrder) ** (1 / iOrder)), T)

    # do the (cascaded) filter process
    for k in range(0, iNumBands):
        X[k, :] = afAudioData
        for j in range(0, 4):
            X[k, :] = lfilter(afCoeffB[j, :, k], afCoeffA[j, :, k], X[k, :])

    return (X)


# see function ERBSpace.m from Slaneys Auditory Toolbox
def getMidFrequencies(f_low, f_hi, iNumBands, fEarQ, fBW):
    freq = np.log((f_low + fEarQ * fBW) / (f_hi + fEarQ * fBW)) / iNumBands
    f_c = np.exp(np.arange(1, iNumBands + 1) * freq)
    f_c = -(fEarQ * fBW) + f_c * (f_hi + fEarQ * fBW)

    return (f_c)


# see function MakeERBFilters.m from Slaneys Auditory Toolbox
def getCoeffs(f_c, B, T):
    fCos = np.cos(2 * f_c * np.pi * T)
    fSin = np.sin(2 * f_c * np.pi * T)
    fExp = np.exp(B * T)
    fSqrtA = 2 * np.sqrt(3 + 2 ** (3 / 2))
    fSqrtS = 2 * np.sqrt(3 - 2 ** (3 / 2))

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * fCos / fExp
    B2 = np.exp(-2 * B * T)

    A11 = -(2 * T * fCos / fExp + fSqrtA * T * fSin / fExp) / 2
    A12 = -(2 * T * fCos / fExp - fSqrtA * T * fSin / fExp) / 2
    A13 = -(2 * T * fCos / fExp + fSqrtS * T * fSin / fExp) / 2
    A14 = -(2 * T * fCos / fExp - fSqrtS * T * fSin / fExp) / 2

    fSqrtA = np.sqrt(3 + 2 ** (3 / 2))
    fSqrtS = np.sqrt(3 - 2 ** (3 / 2))
    fArg = (f_c * np.pi * T) * 1j

    fExp1 = 2 * np.exp(4 * fArg)
    fExp2 = 2 * np.exp(-(B * T) + 2 * fArg)

    afGain = np.abs((-fExp1 * T + fExp2 * T * (fCos - fSqrtS * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos + fSqrtS * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos - fSqrtA * fSin)) *
                    (-fExp1 * T + fExp2 * T * (fCos + fSqrtA * fSin)) /
                    (-2 / np.exp(2 * B * T) - fExp1 + (2 + fExp1) / fExp) ** 4)

    # this is Slaney's compact format - now resort into 3D Matrices
    # fcoefs = [A0*ones(length(f_c),1) A11 A12 A13 A14 A2*ones(length(f_c),1) B0*ones(length(f_c),1) B1 B2 afGain];

    afCoeffB = np.zeros([4, 3, B.size])
    afCoeffA = np.zeros([4, 3, B.size])

    for k in range(0, B.size):
        afCoeffB[0, :, k] = [A0, A11[k], A2] / afGain[k]
        afCoeffA[0, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[1, :, k] = [A0, A12[k], A2]
        afCoeffA[1, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[2, :, k] = [A0, A13[k], A2]
        afCoeffA[2, :, k] = [B0, B1[k], B2[k]]

        afCoeffB[3, :, k] = [A0, A14[k], A2]
        afCoeffA[3, :, k] = [B0, B1[k], B2[k]]

    return (afCoeffB, afCoeffA)


#########################################################################################
# VOICE ACTIVITY DETECTION FUNCTIONS                                                    #
#########################################################################################

def filter_voice(signal, sample_rate, threshold=0.5, mode=3):
    # signal = butter_bandpass_filter(signal, 1500, 5000, sample_rate, 1)
    # signal = np.array(signal, dtype=np.int16)

    signal = np.ascontiguousarray(signal)
    vad = webrtcvad.Vad(mode)
    frames = frame_generator(10, signal, sample_rate)
    frames = list(frames)

    if len(frames) == 0:
        return 0

    match = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            match += 1

    percentage_voice = match * 100 / len(frames)
    return percentage_voice >= threshold


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio) - 1:
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])
