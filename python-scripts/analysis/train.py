import pandas as pd
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal
from CONFIG import *

def sound_location_generator():
    df_dataset = pd.read_csv(PATH_DATASET)
    labels = tf.keras.utils.to_categorical(df_dataset['labels'])
    i = 0
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = labels[i]
        head_position_pan = item['joint2']
        head_position_tilt = item['joint0']
        
        fs, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename, size_chunks=LENGTH_AUDIO)

        for signal1, signal2 in zip(chunks_channel1, chunks_channel2):
            # signal1 = signal.resample(np.array(signal1), RESAMPLING_F)
            # signal2 = signal.resample(np.array(signal2), RESAMPLING_F)

            gamma_sig1 = ToolGammatoneFb(signal1,  RESAMPLING_F, iNumBands=NUM_BANDS)
            gamma_sig2 = ToolGammatoneFb(signal2, RESAMPLING_F, iNumBands=NUM_BANDS)
            input = np.stack((gamma_sig1, gamma_sig2,), axis=2)

            
            # input = np.vstack((signal.resample(np.array(signal1, dtype=float), 6000), signal.resample(np.array(signal2, dtype=float), 6000)))
            # input = concat_fourier_transform(signal1, signal2)
            

            input = gcc_phat(signal1, signal2, RESAMPLING_F)
            input = np.expand_dims(input, axis=-1)
            yield input, np.squeeze(azimuth_location)

        i += 1


def sound_location_format(df_dataset):
    # df_dataset = pd.read_csv("/home/jonas/CLionProjects/soundLocalizer/python-scripts/analysis/output_dataset.csv")

    X = []
    Y = []
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = item['labels']

        fs, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename, size_chunks=LENGTH_AUDIO)

        for signal1, signal2 in zip(chunks_channel1, chunks_channel2):
            signal1 = signal.resample(np.array(signal1), RESAMPLING_F)
            signal2 = signal.resample(np.array(signal2), RESAMPLING_F)

            gamma_sig1 = ToolGammatoneFb(signal1, RESAMPLING_F, iNumBands=NUM_BANDS)
            gamma_sig2 = ToolGammatoneFb(signal2, RESAMPLING_F, iNumBands=NUM_BANDS)
            # input = np.vstack((signal.resample(np.array(signal1, dtype=float), 6000), signal.resample(np.array(signal2, dtype=float), 6000)))
            # input = concat_fourier_transform(signal1, signal2)
            input = np.stack((gamma_sig1, gamma_sig2,), axis=2)
            X.append(input)
            Y.append(azimuth_location)

    return X, Y


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2000),
        tf.keras.callbacks.TensorBoard("data/log"),
    ]




def main(df):
    output_shape = df['labels'].max() + 1


    # X, y = sound_location_format(df)
    # y = tf.keras.utils.to_categorical(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.02, random_state=35)

    # ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X)).batch(BATCH_SIZE)
    # ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(len(X)).batch(BATCH_SIZE)

    ds = tf.data.Dataset.from_generator(
     sound_location_generator,
     (tf.float32, tf.int64),((212993,1), (output_shape))
     ).shuffle(717).batch(BATCH_SIZE)
    #print(next(iter(ds)))


    N_TRAIN = 717
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

    model = get_model_cnn(output_shape=output_shape)


    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        INIT_LR,
        decay_steps=STEPS_PER_EPOCH,
        decay_rate=1,
        staircase=False)

    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # print(f'Training with {len(X_train)} datapoints')
    # print(f'Validation  with {len(X_val)} datapoints')

    model.fit(ds, epochs=EPOCHS, callbacks=get_callbacks())

    # tf_dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    # tf_dataset_test = tf_dataset_test.batch(1)
    # res = model.evaluate(tf_dataset_test, verbose=2)
    # print(res)

    model.save('data/final_model.h5')



if __name__ == '__main__':
    df = pd.read_csv(PATH_DATASET)
    main(df)
