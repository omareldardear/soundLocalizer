import scipy.io.wavfile as wavfile
import webrtcvad
from scipy.signal import filtfilt, butter, lfilter

import collections
import contextlib
import wave
import librosa
import numpy as np
import math

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CONFIG import *


#########################################################################################
# AUDIO PROCESSING FUNCTIONS                                                            #
#########################################################################################


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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
        power += i**2

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


def split_audio_chunks(audio_filename, size_chunks=500, overlap=100):
    fs, signal = wavfile.read(audio_filename, "wb", )

    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    length_chunk = int((fs * size_chunks) / 1000)
    overlap =  int((fs * overlap) / 1000)

    index_start = 0
    chunk_signal1 = []
    chunk_signal2 = []

    while (index_start + length_chunk) < len(signal1):
        chunk_signal1.append(signal1[index_start:index_start + length_chunk])
        chunk_signal2.append(signal2[index_start:index_start + length_chunk])

        index_start += overlap

    pad_sig1 = padarray(signal1[index_start:], length_chunk)
    pad_sig2 = padarray(signal2[index_start:], length_chunk)

    chunk_signal1.append(pad_sig1)
    chunk_signal2.append(pad_sig2)

    return fs, chunk_signal1, chunk_signal2



#########################################################################################
# FEATURES EXTRACTIONS FUNCTIONS                                                        #
#########################################################################################

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


def gcc_phat(sig, refsig, fs=1, max_tau=0.00040, interp=1, n_delay=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(n // 2 + 1)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))

    delay = []

    # find max cross correlation index
    shift = np.argmax(cc) - max_shift

    tau = shift / float(interp * fs)

    return [tau], cc


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

def filter_voice(signal, sample_rate, mode=3):
    signal = butter_bandpass_filter(signal, 1500, 5000, sample_rate, 1)
    signal = np.array(signal, dtype=np.int16)

    signal = np.ascontiguousarray(signal)
    vad = webrtcvad.Vad(mode)
    frames = frame_generator(10, signal, sample_rate)
    frames = list(frames)

    match = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, 48000)
        if is_speech:
            match += 1

    percentage_voice = match * 100 / len(frames)
    return percentage_voice > THRESHOLD_VOICE


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
    while offset + n < len(audio) -1:
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



