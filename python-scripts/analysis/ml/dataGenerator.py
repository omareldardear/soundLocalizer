import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wavfile
import scipy.signal

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import gcc_phat, butter_lowpass_filter
from CONFIG import *

import pickle
import os
import librosa
from gammatone.fftweight import fft_gtgram


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataFrame, data_directory, features, num_classe, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 resample=0, shuffle=True,):

        'Initialization'
        self.df = dataFrame
        self.dim = dim
        self.batch_size = batch_size
        self.get_labels()
        self.list_IDs = dataFrame['audio_filename']
        self.n_channels = n_channels
        self.n_classes = num_classe
        self.shuffle = shuffle
        self.on_epoch_end()
        self.resampling = resample
        self.features = features
        self.path_data = data_directory

        self.max_tau = DISTANCE_MIC / 343.2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_labels(self):
        self.labels = {}

        for index, item in self.df.iterrows():
            self.labels[item['audio_filename']] = item['labels']

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float)
        y = np.empty(self.batch_size, dtype=np.float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if self.features == 'gammatone':
                filename = os.path.join(self.path_data, ID.split('.wav')[0])

                input_x = pickle.load(open(filename, 'rb'))
                input_x = input_x.reshape(input_x.shape[1], input_x.shape[0])
                input_x = np.expand_dims(input_x, axis=-1)

            else:
                filename = os.path.join(self.path_data, ID)

                fs, signal = wavfile.read(filename, "wb")
                signal1 = signal[:, 0]
                signal2 = signal[:, 1]

                if self.resampling:
                    nb_samples = self.resampling
                    signal1 = np.array(scipy.signal.resample(signal1, nb_samples), dtype=np.int16)
                    signal2 = np.array(scipy.signal.resample(signal2, nb_samples), dtype=np.int16)
                    fs = nb_samples

                if self.features == 'gcc-phat':
                    window_hanning = np.hanning(len(signal1))
                    delay, gcc = gcc_phat(signal1 * window_hanning, signal2 * window_hanning, fs, self.max_tau)
                    input_x = np.expand_dims(gcc, axis=-1)

                elif self.features == 'melspec':
                    input_x = librosa.feature.melspectrogram(signal1, fs)
                    input_x = np.expand_dims(input_x, axis=-1)

                elif self.features == 'gammagram':
                    twin = 0.08
                    thop = twin / 2
                    channels = 64
                    fmin = 20
                    signal = signal.mean(1)

                    signal1 = fft_gtgram(signal1, fs, twin, thop, channels, fmin)
                    signal2 = fft_gtgram(signal2, fs, twin, thop, channels, fmin)

                    input_x = np.stack((signal1, signal2), axis=-1)



                else:
                    signal1 = butter_lowpass_filter(signal1, 1000, fs)
                    signal2 = butter_lowpass_filter(signal2, 1000, fs)

                    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
                    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

                    input_x = np.stack((signal1, signal2), axis=-1)

            # Store sample
            X[i,] = input_x

            # Store class
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGenerator_headPose(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, dataFrame, data_directory, features, num_classe, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 resample=0, shuffle=True):

        'Initialization'
        self.df = dataFrame
        self.dim = dim
        self.batch_size = batch_size
        self.get_labels()
        self.list_IDs = dataFrame['audio_filename']
        self.n_channels = n_channels
        self.n_classes = num_classe
        self.shuffle = shuffle
        self.on_epoch_end()
        self.resampling = resample
        self.features = features
        self.path_data = data_directory

        self.max_tau = DISTANCE_MIC / 343.2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, X_head, y = self.__data_generation(list_IDs_temp)

        return [X, X_head], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_labels(self):
        self.labels = {}

        for index, item in self.df.iterrows():
            self.labels[item['audio_filename']] = item['labels']

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float)
        X_head = np.empty((self.batch_size, 2, 1), dtype=np.float)
        y = np.empty(self.batch_size, dtype=np.float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if self.features == 'gammatone':
                filename = os.path.join(self.path_data, ID.split('.wav')[0])

                input_x = pickle.load(open(filename, 'rb'))
                input_x = input_x.reshape(input_x.shape[1], input_x.shape[0])
                input_x = np.expand_dims(input_x, axis=-1)

            else:
                filename = os.path.join(self.path_data, ID)

                fs, signal = wavfile.read(filename, "wb")
                signal1 = signal[:, 0]
                signal2 = signal[:, 1]

                if self.resampling:
                    nb_samples = self.resampling
                    signal1 = np.array(scipy.signal.resample(signal1, nb_samples), dtype=np.int16)
                    signal2 = np.array(scipy.signal.resample(signal2, nb_samples), dtype=np.int16)
                    fs = nb_samples

                if self.features == 'gcc-phat':
                    window_hanning = np.hanning(len(signal1))
                    delay, gcc = gcc_phat(signal1 * window_hanning, signal2 * window_hanning, fs, self.max_tau)
                    input_x = np.expand_dims(gcc, axis=-1)

                elif self.features == 'melspec':
                    input_x = librosa.feature.melspectrogram(signal1, fs)
                    input_x = np.expand_dims(input_x, axis=-1)

                elif self.features == 'gammagram':

                    filename = os.path.join(self.path_data, ID.split('.wav')[0])

                    if os.path.exists(filename):
                        input_x = pickle.load(open(filename, 'rb'))
                    else:

                        twin = 0.08
                        thop = twin / 2
                        channels = 64
                        fmin = 20

                        signal1 = fft_gtgram(signal1, fs, twin, thop, channels, fmin)
                        signal2 = fft_gtgram(signal2, fs, twin, thop, channels, fmin)

                        input_x = np.stack((signal1, signal2), axis=-1)
                        pickle.dump(input_x, open(filename, "wb"))

                else:
                    signal1 = butter_lowpass_filter(signal1, 1000, fs)
                    signal2 = butter_lowpass_filter(signal2, 1000, fs)

                    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
                    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

                    input_x = np.stack((signal1, signal2), axis=-1)

            # Store sample
            X[i,] = input_x

            # Store head position
            X_head[i,] = np.array([self.df[self.df['audio_filename'] == ID]['joint0'].values, self.df[self.df['audio_filename'] == ID]['joint2'].values])

            # Store class
            y[i] = self.labels[ID]

        return X, X_head, y
