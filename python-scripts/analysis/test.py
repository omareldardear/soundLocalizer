import pandas as pd
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal
from CONFIG import *
import os


def sound_location_generator(df_dataset, labels):
    i = 0
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = labels[index]
        head_position_pan = item['joint2']
        head_position_tilt = item['joint0']

        fs, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename, size_chunks=LENGTH_AUDIO)

        for signal1, signal2 in zip(chunks_channel1, chunks_channel2):
            signal1 = signal.resample(np.array(signal1), RESAMPLING_F)
            signal2 = signal.resample(np.array(signal2), RESAMPLING_F)

            # gamma_sig1 = ToolGammatoneFb(signal1, RESAMPLING_F, iNumBands=NUM_BANDS)
            # gamma_sig2 = ToolGammatoneFb(signal2, RESAMPLING_F, iNumBands=NUM_BANDS)
            # input = np.stack((gamma_sig1, gamma_sig2,), axis=2)

            # input = np.vstack((signal.resample(np.array(signal1, dtype=float), 6000), signal.resample(np.array(signal2, dtype=float), 6000)))
            # input = concat_fourier_transform(signal1, signal2)

            input = gcc_phat(signal1, signal2, RESAMPLING_F)
            input = np.concatenate((input, [head_position_pan, head_position_tilt]))
            input = np.expand_dims(input, axis=-1)

            yield input, np.squeeze(azimuth_location)

        i += 1


def main(df):
    output_shape = df['labels'].max() + 1

    # Create a new model instance
    model = get_model_1dcnn(output_shape)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(INIT_LR, decay=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # Load the previously saved weights
    latest = 'training_2/cp-0020.ckpt'
    model.load_weights(latest)

    df_test = df[df['subject_id'].isin(TEST_SUBJECTS)]
    labels = tf.keras.utils.to_categorical(df['labels'])



    ds_test = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_test, labels),
        (tf.float32, tf.int64), ((None, 1), output_shape)
    ).shuffle(100).batch(BATCH_SIZE)

    # Re-evaluate the model
    los, acc = model.evaluate(ds_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('data/saved_model/my_model')

    return 1


if __name__ == "__main__":
    df = pd.read_csv(PATH_DATASET)

    main(df)
