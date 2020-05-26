import pandas as pd
from utils import gcc_phat, split_audio_chunks, concat_fourier_transform, ToolGammatoneFb
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal

BATCH_SIZE = 8
EPOCHS = 200
RESAMPLING_F = 16000
INIT_LR = 1e-3


def sound_location_generator():
    df_dataset = pd.read_csv("/home/jonas/CLionProjects/soundLocalizer/python-scripts/analysis/output_dataset.csv")

    i = 0
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = item['labels']

        fs, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename)

        for signal1, signal2 in zip(chunks_channel1, chunks_channel2):
            # cc = gcc_phat(signal1, signal2)
            signal1 = signal.resample(np.array(signal1, dtype=float), RESAMPLING_F)
            signal2 = signal.resample(np.array(signal2, dtype=float), RESAMPLING_F)

            gamma_sig1 = ToolGammatoneFb(signal1, RESAMPLING_F)
            gamma_sig2 = ToolGammatoneFb(signal2, RESAMPLING_F)
            # input = np.vstack((signal.resample(np.array(signal1, dtype=float), 6000), signal.resample(np.array(signal2, dtype=float), 6000)))
            # input = concat_fourier_transform(signal1, signal2)
            input = np.stack((gamma_sig1, gamma_sig2,), axis=2)
            yield input, azimuth_location

            i += 1


def sound_location_format(df_dataset):
    # df_dataset = pd.read_csv("/home/jonas/CLionProjects/soundLocalizer/python-scripts/analysis/output_dataset.csv")

    X = []
    Y = []
    for index, item in df_dataset.iterrows():
        audio_filename = item['audio_filename']
        azimuth_location = item['labels']

        fs, chunks_channel1, chunks_channel2 = split_audio_chunks(audio_filename)

        for signal1, signal2 in zip(chunks_channel1, chunks_channel2):
            signal1 = signal.resample(np.array(signal1, dtype=float), RESAMPLING_F)
            signal2 = signal.resample(np.array(signal2, dtype=float), RESAMPLING_F)

            gamma_sig1 = ToolGammatoneFb(signal1, RESAMPLING_F)
            gamma_sig2 = ToolGammatoneFb(signal2, RESAMPLING_F)
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


def get_model(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=50, kernel_size=(7, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 1)),

        tf.keras.layers.Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1, 1)),

        tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1, 1)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def main(df):
    output_shape = df['labels'].max() + 1
    X, y = sound_location_format(df)
    y = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.02, random_state=35)

    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X)).batch(BATCH_SIZE)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(len(X)).batch(BATCH_SIZE)

    N_TRAIN = len(X_train)
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

    model = get_model(output_shape=output_shape)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        INIT_LR,
        decay_steps=STEPS_PER_EPOCH,
        decay_rate=1,
        staircase=False)

    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f'Training with {len(X_train)} datapoints')
    print(f'Validation  with {len(X_val)} datapoints')

    model.fit(ds, epochs=EPOCHS, callbacks=get_callbacks(), validation_data=ds_val)

    tf_dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    tf_dataset_test = tf_dataset_test.batch(1)
    res = model.evaluate(tf_dataset_test, verbose=2)

    model.save('data/final_model.h5')

    print(res)


if __name__ == '__main__':
    df = pd.read_csv('/home/jonas/CLionProjects/soundLocalizer/python-scripts/analysis/output_dataset.csv')
    #
    # test_filename = df.iloc[0]['audio_filename']
    # fs, sig1, sig2 = split_audio_chunks(test_filename)
    #
    # cc = gcc_phat(sig1[1], sig2[1])
    #
    # plt.plot(cc)
    # plt.show()

    main(df)
