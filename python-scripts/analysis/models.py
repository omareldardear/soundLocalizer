import tensorflow as tf
from tensorflow.keras.backend import squeeze


#########################################################################################
#                               MODELS DEFINITION                                       #
#########################################################################################

def get_model_cnn(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=50, kernel_size=(7, 2), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 1)),

        tf.keras.layers.Conv2D(filters=60, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1, 1)),

        tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 1)),

        # tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same',
        #                        kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D((2, 1)),

        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

        # tf.keras.layers.Reshape((-1, 60)),

        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def get_model_dense(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def get_model_1dcnn(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(3),

        tf.keras.layers.Dropout(rate=0.4),

        tf.keras.layers.Reshape((-1, 128)),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(rate=0.4),

        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def simple_lstm(output_dim):
    model = tf.keras.Sequential()

    # Add a LSTM layer with 128 internal units.

    model.add(tf.keras.layers.Conv1D(filters=90, kernel_size=7, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=7, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0005)))

    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Reshape((-1, 50)))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dropout(rate=1 - 0.5))

    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

    return model


def conv_net_lstm_attention(output_dim=11, mel_dim=(32, 26, 1)):
    inputs = tf.keras.layers.Input((mel_dim))

    x = tf.keras.layers.Conv2D(filters=10, kernel_size=(5, 1), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=40, kernel_size=(5, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, 16])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(128)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32)(x)
    output = tf.keras.layers.Dense(output_dim, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[output])

    return model