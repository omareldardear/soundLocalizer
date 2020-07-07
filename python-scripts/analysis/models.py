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
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((1, 1)),

        tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 1)),

        tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 1)),

        tf.keras.layers.Reshape((-1, 90)),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def get_model_dense(input_shape, output_shape):
    inputs = tf.keras.layers.Input(input_shape)
    x_headPosition = tf.keras.layers.Input((2, 1))

    x = tf.keras.layers.Lambda(lambda q: tf.concat([q, x_headPosition], axis=1))(inputs)
    x = tf.keras.layers.Lambda(lambda q: tf.expand_dims(q, axis=-1))(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    output = tf.keras.layers.Dense(output_shape, activation="softmax")(x)

    model = tf.keras.Model(inputs=[inputs, x_headPosition], outputs=[output])

    return model


def get_model_dense_simple( output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(output_shape, activation="softmax"),

    ])

    return model




def get_model_1dcnn_simple(output_shape):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=512, kernel_size=15, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(8),

        tf.keras.layers.Conv1D(filters=256, kernel_size=11, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(4),

        tf.keras.layers.Conv1D(filters=128, kernel_size=9, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(4),

        tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),


        # tf.keras.layers.Reshape((-1, 128)),

        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        #
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
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


def conv1d_net_lstm_attention(input_shape, output_dim=11):
    inputs = tf.keras.layers.Input(input_shape)
    x_headPosition = tf.keras.layers.Input((2, 1))

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=11, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPooling1D(4)(x)

    x = tf.keras.layers.Conv1D(filters=90, kernel_size=9, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPooling1D(4)(x)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)


    # x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.Lambda(lambda q: squeeze(q, -1), name='squeeze_last_dim') (x)
    x = tf.keras.layers.Reshape((-1, 64))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, 128])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(256)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(256, activation='relu')(attVector)
    # x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Lambda(lambda q: tf.concat([q, x_headPosition[:, :, 0]], axis=1))(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(output_dim, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=[inputs, x_headPosition], outputs=[output])

    return model


def conv_net_lstm_attention(input_shape, output_dim=11):

    inputs = tf.keras.layers.Input(input_shape)
    x_headPosition = tf.keras.layers.Input((2, 1))

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 2), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=90, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Reshape((-1, 64))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, 128])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(256)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(256, activation='relu')(attVector)
    # x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Lambda(lambda q: tf.concat([q, x_headPosition[:, :, 0]], axis=1))(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(output_dim, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=[inputs, x_headPosition], outputs=[output])

    return model

