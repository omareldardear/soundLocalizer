import pandas as pd
from utils import *
import tensorflow as tf
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import soundfile as sf
from CONFIG import *
import os, io

random_state = 42


def sound_location_generator(df_dataset, labels, features='gcc-phat'):
    i = 0
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = labels[index]
        head_position_pan = item['joint2']
        head_position_tilt = item['joint0']

        sample_rate, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename, size_chunks=LENGTH_AUDIO)

        for signal1, signal2 in zip(chunks_channel1, chunks_channel2):
            number_of_samples = round(len(signal1) * float(RESAMPLING_F) / sample_rate)
            signal1 = np.array(signal.resample(signal1, number_of_samples), dtype=np.float32)
            signal2 = np.array(signal.resample(signal2, number_of_samples), dtype=np.float32)

            if features == 'gcc-phat':

                input = gcc_phat(signal1, signal2, RESAMPLING_F)
                input = np.concatenate((input, [head_position_pan, head_position_tilt]))
                input = np.expand_dims(input, axis=-1)

            elif features == 'gammatone':

                input = gcc_gammatoneFilter(signal1, signal2, RESAMPLING_F, NUM_BANDS)
                input = np.expand_dims(input, axis=-1)

            # input = np.vstack((signal.resample(np.array(signal1, dtype=float), 6000), signal.resample(np.array(signal2, dtype=float), 6000)))
            # input = concat_fourier_transform(signal1, signal2)

            yield input, np.squeeze(azimuth_location)

        i += 1


def get_callbacks():
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "/tmp/training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    return [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2000),
        tf.keras.callbacks.TensorBoard("/tmp/data/log"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=20)
    ]


def get_generator_dataset(df, output_shape):
    labels = tf.keras.utils.to_categorical(df['labels'])

    df_test = df[df['subject_id'].isin(TEST_SUBJECTS)]
    df = df.drop(df_test.index)

    ds_train = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df, labels, FEATURE),
        (tf.float32, tf.int64), ((None, 1), output_shape)
    ).shuffle(1000).batch(BATCH_SIZE)

    ds_test = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_test, labels, FEATURE),
        (tf.float32, tf.int64), ((None, 1), output_shape)
    ).shuffle(100).batch(BATCH_SIZE)

    return ds_train, ds_test


def main(df):
    output_shape = df['labels'].max() + 1

    ds_train, ds_test = get_generator_dataset(df, output_shape)

    model = get_model_dense(output_shape=output_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(ds_train, epochs=EPOCHS, callbacks=get_callbacks())

    # Re-evaluate the model
    los, acc = model.evaluate(ds_test, verbose=2)
    print("Test model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('/tmp/data/saved_model/my_model')


if __name__ == '__main__':
    df = pd.read_csv(PATH_DATASET)
    main(df)
