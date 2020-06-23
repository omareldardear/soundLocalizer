import pandas as pd
from utils import *
import numpy as np
import scipy.signal
from CONFIG import *
from models import *
import tensorflow as tf

random_state = 42
max_tau = DISTANCE_MIC / 343.2


def sound_location_generator(df_dataset, labels, features='gcc-phat'):
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = labels[index]
        head_position_pan = item['joint2']
        head_position_tilt = item['joint0']

        fs, signal = wavfile.read(audio_filename, "wb")
        signal1 = signal[:, 0]
        signal2 = signal[:, 1]

        number_of_samples = round(len(signal1) * float(RESAMPLING_F) / fs)
        signal1 = np.array(scipy.signal.resample(signal1, number_of_samples), dtype=np.float32)
        signal2 = np.array(scipy.signal.resample(signal2, number_of_samples), dtype=np.float32)

        if features == 'gcc-phat':
            window_hanning = np.hanning(number_of_samples)
            input_x = gcc_phat(signal1 * window_hanning, signal2 * window_hanning, RESAMPLING_F, max_tau)
            norm = np.linalg.norm(input_x)
            input_x = input_x / norm
            input_x = np.concatenate((input_x, [head_position_pan, head_position_tilt]))
            input_x = np.expand_dims(input_x, axis=-1)

        elif features == 'gammatone':
            input_x = gcc_gammatoneFilter(signal1, signal2, RESAMPLING_F, NUM_BANDS)
            input_x = input_x.reshape(input_x.shape[1], input_x.shape[0])
            input_x = np.expand_dims(input_x, axis=-1)

        elif features == 'mfcc':
            signal1 = get_MFCC(signal1, RESAMPLING_F)
            signal2 = get_MFCC(signal2, RESAMPLING_F)
            input_x = np.stack((signal1, signal2), axis=2)

        else:
            # signal1 = np.concatenate((signal1, [head_position_pan, head_position_tilt]))
            # signal2 = np.concatenate((signal2, [head_position_pan, head_position_tilt]))
            input_x = np.stack((signal1, signal2), axis=-1)

        yield input_x, np.squeeze(azimuth_location)


def get_callbacks():
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "/tmp/training_2/cp-{epoch:04d}.ckpt"

    return [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2000),
        tf.keras.callbacks.TensorBoard("data/log"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=20)
    ]


def get_generator_dataset(df_input, output_shape):
    labels = tf.keras.utils.to_categorical(df_input['labels'])

    df_test = df_input[df_input['subject_id'].isin(TEST_SUBJECTS)]
    df_train = df_input.drop(df_test.index)

    ds_train = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_train, labels, FEATURE),
        (tf.float32, tf.int64), ((None, 2), output_shape))

    ds_test = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_test, labels, FEATURE),
        (tf.float32, tf.int64), ((None, 1), output_shape))



    return ds_train, ds_test


def main(df):
    output_shape = int(df['labels'].max() + 1)

    ds_train, ds_test = get_generator_dataset(df, output_shape)


    model = conv_net_lstm_attention(output_shape, (26,52,1))

    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ds_train = ds_train.shuffle(df.shape[0]).batch(BATCH_SIZE)

    model.fit(ds_train, callbacks=get_callbacks(), epochs=EPOCHS)

    # Re-evaluate the model
    los, acc = model.evaluate(ds_test, verbose=2)
    print("Test model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('/tmp/data/saved_model/my_model')


if __name__ == '__main__':
    df = pd.read_csv(PATH_DATASET)
    main(df)
