import pandas as pd
from utils_ml import *

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CONFIG import *

from models import *
import tensorflow as tf
from dataGenerator import DataGenerator, DataGenerator_headPose
import argparse
from sklearn import preprocessing

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



def main(df_input, args):
    output_shape = int(df_input['labels'].max() + 1)

    features_to_normalize = ['joint2', 'joint0']
    df[features_to_normalize] = df_input[features_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Model parameters
    params = {'dim': INPUT_SHAPE,
              'batch_size': BATCH_SIZE,
              'n_channels': NB_CHANNELS,
              'resample': RESAMPLING_F,
              'shuffle': True,
              'reg': args.regression}

    params_test = {'dim': INPUT_SHAPE,
              'batch_size': 4,
              'n_channels': NB_CHANNELS,
              'resample': RESAMPLING_F,
              'shuffle': False,
              'reg': args.regression}

    # Define train and test generators
    df_train, df_test = get_datasets(df_input, TEST_SUBJECTS)

    training_generator = DataGenerator(df_train, PATH_DATA, FEATURE, output_shape, **params)
    test_generator = DataGenerator(df_test, PATH_DATA, FEATURE, output_shape, **params_test)

    # Define the model
    model = get_model_cnn(output_shape, reg=args.regression)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INIT_LR,
        decay_steps=1500,
        decay_rate=0.94,
        name="lr_decay"
    )

    optimizer_obj = tf.keras.optimizers.Adam(INIT_LR)

    if args.regression:
        model.compile(optimizer=optimizer_obj,
                      loss='mse',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    model.fit(training_generator, callbacks=get_callbacks(m_patience=40), epochs=EPOCHS, validation_data=test_generator)

    model.summary()


    # Re-evaluate the model
    model.evaluate(test_generator, verbose=2)
    model.save(os.path.join(SAVED_MODEL_PATH, 'my_model.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--azimuth_resolution", type=int, default=1,
                        help="Angle resolution for azimuth")

    parser.add_argument("--regression", action='store_true',
                        help="Regression or classification")

    parser_args = parser.parse_args()
    df = pd.read_csv(PATH_DATASET)

    df['labels'] = (df['azimuth'] + 90)

    if  parser_args.regression:
        df['labels'] = df['labels'] / 180
        df['labels'] = round(df['labels'], 2)

    elif parser_args.azimuth_resolution:
        df['labels'] = df['labels'] // parser_args.azimuth_resolution

    main(df, parser_args)
