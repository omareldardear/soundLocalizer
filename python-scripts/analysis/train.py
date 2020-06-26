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

        input_head = np.array([head_position_pan, head_position_tilt])
        input_head = np.expand_dims(input_head, axis=-1)

        if features == 'gcc-phat':
            window_hanning = np.hanning(number_of_samples)
            input_x = gcc_phat(signal1 * window_hanning, signal2 * window_hanning, RESAMPLING_F, max_tau)
            # norm = np.linalg.norm(input_x)
            # input_x = input_x / norm
            input_x = np.expand_dims(input_x, axis=-1)

            yield {"input_1": input_x, "input_2": input_head}, np.squeeze(azimuth_location)

        elif features == 'gammatone':
            input_x = gcc_gammatoneFilter(signal1, signal2, RESAMPLING_F, NUM_BANDS)
            input_x = input_x.reshape(input_x.shape[1], input_x.shape[0])
            input_x = np.expand_dims(input_x, axis=-1)

            yield {"input_1": input_x, "input_2": input_head}, np.squeeze(azimuth_location)

        elif features == 'mfcc':
            signal1 = get_MFCC(signal1, RESAMPLING_F)
            signal2 = get_MFCC(signal2, RESAMPLING_F)
            input_x = np.stack((signal1, signal2), axis=2)

            yield input_x, np.squeeze(azimuth_location)


        else:
            # signal1 = np.concatenate((signal1, [head_position_pan, head_position_tilt]))
            # signal2 = np.concatenate((signal2, [head_position_pan, head_position_tilt]))
            input_x = np.stack((signal1, signal2), axis=-1)

            yield {"input_1": input_x, "input_2": input_head}, np.squeeze(azimuth_location)


def get_callbacks():
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "/tmp/training_2/cp-{epoch:04d}.ckpt"

    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
        tf.keras.callbacks.TensorBoard("data/log"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=20)
    ]


def get_generator_dataset(df_input, output_dim):
    labels = tf.keras.utils.to_categorical(df_input['labels'])

    df_test = df_input[df_input['subject_id'].isin(TEST_SUBJECTS)]
    df_train = df_input.drop(df_test.index)

    df_val = df_train.sample(frac=0.1)
    df_train = df_train.drop(df_val.index)

    ds_train = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_train, labels, FEATURE),
        output_types=({"input_1": tf.int64, "input_2": tf.int64}, tf.int64),
        output_shapes=({"input_1": INPUT_SHAPE, "input_2": (None, 1)}, output_dim)
    ).shuffle(df_train.shape[0] // 2)

    ds_val = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_val, labels, FEATURE),
        output_types=({"input_1": tf.int64, "input_2": tf.int64}, tf.int64),
        output_shapes=({"input_1": INPUT_SHAPE, "input_2": (None, 1)}, output_dim)
    ).batch(32)

    ds_test = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_test, labels, FEATURE),
        output_types=({"input_1": tf.int64, "input_2": tf.int64}, tf.int64),
        output_shapes=({"input_1": INPUT_SHAPE, "input_2": (None, 1)}, output_dim)
    ).batch(32)

    return ds_train, ds_val, ds_test


def main(df):
    output_shape = int(df['labels'].max() + 1)

    features_to_normalize = ['joint2', 'joint0']
    df[features_to_normalize] = df[features_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    ds_train, ds_val, ds_test = get_generator_dataset(df, output_shape)

    model = get_model_dense(INPUT_SHAPE, output_shape)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INIT_LR,
        decay_steps=1000,
        decay_rate=0.94,
        name="lr_decay"
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)

    ds_train = ds_train.batch(BATCH_SIZE)

    model.fit(ds_train, callbacks=get_callbacks(), epochs=EPOCHS, validation_data=ds_val)

    # Re-evaluate the model
    los, acc = model.evaluate(ds_test, verbose=2)
    print("Test model, accuracy: {:5.2f}%".format(100 * acc))
    tf.saved_model.save(model, '/tmp/data/saved_model/my_model.h5')


if __name__ == '__main__':
    df = pd.read_csv(PATH_DATASET)
    df = df.sample(frac=1).reset_index(drop=True)
    main(df)
