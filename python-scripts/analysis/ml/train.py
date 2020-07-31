import pandas as pd
from utils_ml import *

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CONFIG import *

from models import *
import tensorflow as tf
from dataGenerator import DataGenerator
import argparse

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




def main(df_input):
    output_shape = int(df_input['labels'].max() + 1)

    features_to_normalize = ['joint2', 'joint0']
    df[features_to_normalize] = df_input[features_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Model parameters
    params = {'dim': INPUT_SHAPE,
              'batch_size': BATCH_SIZE,
              'n_channels': NB_CHANNELS,
              'resample': RESAMPLING_F,
              'shuffle': True}

    # Define train and test generators
    df_train, df_test = get_datasets(df_input, TEST_SUBJECTS)

    training_generator = DataGenerator(df_train, PATH_DATA, FEATURE, output_shape, **params)
    test_generator = DataGenerator(df_test, PATH_DATA, FEATURE, output_shape, **params)

    # Define the model
    model = get_model_cnn(output_shape)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INIT_LR,
        decay_steps=800,
        decay_rate=0.94,
        name="lr_decay"
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    model.fit(training_generator, callbacks=get_callbacks(), epochs=EPOCHS, validation_data=test_generator)

    print(model.summary())


    # Re-evaluate the model
    los, acc = model.evaluate(test_generator, verbose=2)
    print("Test model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('/tmp/data/saved_model/my_model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--azimuth_resolution", type=int, default=0,
                        help="Angle resolution for azimuth")


    parser_args = parser.parse_args()
    df = pd.read_csv(PATH_DATASET)

    if parser_args.azimuth_resolution:
        df['labels'] = (df['azimuth'] + 90)
        df['labels'] = df['labels'] // parser_args.azimuth_resolution

    df = df.sample(frac=1).reset_index(drop=True)
    main(df)
