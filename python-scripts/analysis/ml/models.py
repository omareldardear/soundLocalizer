import tensorflow as tf
from tensorflow.keras.backend import squeeze


#########################################################################################
#                               MODELS ONE INPUT                                        #
#########################################################################################

def get_model_cnn(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    return model


def get_model_dense_simple(output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(2048, activation="relu"),
        # tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(output_shape, activation="softmax"),

    ])

    return model


def get_model_1dcnn_simple(output_shape):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=96, kernel_size=7, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),

        tf.keras.layers.MaxPooling1D(7),

        tf.keras.layers.Conv1D(filters=96, kernel_size=7, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(7),

        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(5),

        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(5),

        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        # tf.keras.layers.BatchNormalization(),

        # tf.keras.layers.Reshape((-1, 128)),

        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        #
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
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


#########################################################################################
#                               MODELS TWO INPUTS                                       #
#########################################################################################


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


def get_model_head_cnn(input_shape, output_dim=11, reg=False):

    # Model for audio
    inputs = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(name='conv1', filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(inputs)
    x = tf.keras.layers.Conv2D(name='conv2', filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.Model(inputs=inputs, outputs=x)

    # Model for head position
    input_head = tf.keras.layers.Input((2, 1))
    y = squeezed = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2))(input_head)

    y = tf.keras.Model(inputs=input_head, outputs=y)

    # combine the output of the two branches
    combined = tf.keras.layers.concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = tf.keras.layers.Dense(512, activation='relu')(combined)

    if reg:
        output = tf.keras.layers.Dense(1, activation="linear")(z)
    else:
        output = tf.keras.layers.Dense(output_dim, activation="softmax", name='output')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = tf.keras.Model(inputs=[x.input, y.input], outputs=output)

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
