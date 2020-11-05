import pandas as pd
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from CONFIG import *
import pickle
from tensorflow.keras.models import load_model
import argparse
import tensorflow as tf
import numpy as np
from utils import get_fft_gram
import scipy.io.wavfile as wavfile
from ml.dataGenerator import DataGenerator

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_datasets(df_input, val=False):
    df_test = df_input[df_input['subject_id'].isin([43])]
    df_train = df_input.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if val:
        df_val = df_train.sample(frac=0.1, random_state=42)
        df_train = df_train.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        return df_train, df_val, df_train

    return df_train, df_test


def error_by_angles(model, df_test):
    predictions = {}

    for _, item in df_test.iterrows():
        tmp = item['audio_filename']
        filename = os.path.join(PATH_DATA, tmp)

        fs, signal = wavfile.read(filename)

        fft_gram1, fft_gram2 = get_fft_gram(signal, fs)
        input_x = np.stack((fft_gram1, fft_gram2), axis=-1)

        input_x = np.expand_dims(input_x, axis=0)

        y_pred = model.predict(input_x)
        y_pred = ((y_pred * 180))[0][0] - 90

        if item['azimuth'] in predictions.keys():
            predictions[item['azimuth']].append(y_pred)
        else:
            predictions[item['azimuth']] = [y_pred]

    return pd.DataFrame.from_dict(predictions, orient='index')


def main(df_input, args):
    model = load_model(args.model_path, custom_objects={'tf': tf})

    # Define train and test generators
    # _, df_test = get_datasets(df_input)

    model.summary()

    params_test = {'dim': INPUT_SHAPE,
                   'batch_size': 4,
                   'n_channels': NB_CHANNELS,
                   'resample': RESAMPLING_F,
                   'shuffle': False,
                   'reg': True}

    df_input['labels'] = (df_input['azimuth'] + 90 ) / 180
    df_input['labels'] = round(df_input['labels'], 2)
    df_input['azimuth'] = df_input['azimuth'] + 90

    test_generator = DataGenerator(df_input, PATH_DATA, FEATURE, 1, **params_test)

    model.evaluate(test_generator, verbose=2)

    df_predictions = error_by_angles(model, df_input)

    df_predictions.head()
    df_predictions['mean'] = df_predictions.mean(axis=1)
    df_predictions.to_csv("predictions_tv.csv")

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Model path (*.h5)")

    parser.add_argument("dataset", type=str,
                        help="Dataset path (*.csv)")

    parser_args = parser.parse_args()

    df = pd.read_csv(parser_args.dataset)

    main(df, parser_args)
