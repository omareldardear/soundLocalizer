import scipy.io.wavfile as wavfile
import numpy as np
from scipy.signal import lfilter
import tensorflow as tf


#########################################################################################
# FEATURES EXTRACTIONS FUNCTIONS                                                        #
#########################################################################################
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=26):
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

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return cc


def concat_fourier_transform(sig1, sig2, n=512):
    # Generalized Cross Correlation Phase Transform
    fft_sig1 = np.fft.fft(sig1, n=n)
    fft_sig2 = np.fft.fft(sig2, n=n)

    fft_phase = np.angle(fft_sig1)[int(len(fft_sig1) / 2):]
    fft_phase2 = np.angle(fft_sig2)[int(len(fft_sig2) / 2):]

    stack_fft = np.vstack((np.array(fft_phase), np.array(fft_phase2)))

    return stack_fft


def split_audio_chunks(audio_filename, size_chunks=500):
    fs, signal = wavfile.read(audio_filename, "wb", )

    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    length_chunk = int((fs * size_chunks) / 1000)

    index_start = 0

    chunk_signal1 = []
    chunk_signal2 = []

    while (index_start + length_chunk) < len(signal1):
        chunk_signal1.append(signal1[index_start:index_start + length_chunk])
        chunk_signal2.append(signal2[index_start:index_start + length_chunk])

        index_start += length_chunk

    return fs, chunk_signal1, chunk_signal2


# see function mfcc.m from Slaneys Auditory Toolbox (Matlab)
def ToolGammatoneFb(afAudioData, f_s, iNumBands=26, f_low=100):
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
# MODELS DEFINITION                                                                     #
#########################################################################################

def get_model_cnn(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=50, kernel_size=(7, 2), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 1)),

        tf.keras.layers.Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1, 1)),

        tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1, 1)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def get_model_dense(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.Dense(2000, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(150, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def get_model_1dcnn(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=50, kernel_size=7, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(filters=60, kernel_size=5, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(3),

        tf.keras.layers.Conv1D(filters=90, kernel_size=3, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model
