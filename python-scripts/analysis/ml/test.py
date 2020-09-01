import pandas as pd
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from CONFIG import *
from models import *
from dataGenerator import DataGenerator
import argparse



def get_datasets(df_input, val=False):
    df_test = df_input[df_input['subject_id'].isin([61])]
    df_train = df_input.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if val:
        df_val = df_train.sample(frac=0.1, random_state=random_state)
        df_train = df_train.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        return df_train, df_val, df_train

    return df_train, df_test[:50]

def main(df_input):
    output_shape = int(df_input['labels'].max() + 1)

    # Create a new model instance
    model = get_model_cnn(output_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])

    # Model parameters
    params = {'dim': INPUT_SHAPE,
              'batch_size': 1,
              'n_channels': 1,
              'resample': RESAMPLING_F,
              'shuffle': True}


    # Load the previously saved weights
    latest = '/tmp/training_2/cp-0020.ckpt'
    model.load_weights(latest)


    # Define train and test generators
    _, df_test = get_datasets(df_input)

    test_generator = DataGenerator(df_test, PATH_DATA, FEATURE, output_shape, **params)

    # Re-evaluate the model
    los, acc = model.evaluate(test_generator, verbose=2)
    print("Restored model, accuracy: {:5.2f%".format( acc))

    return 1


if __name__ == "__main__":
    df = pd.read_csv(PATH_DATASET)
    parser = argparse.ArgumentParser()
    parser.add_argument("--azimuth_resolution", type=int, default=-1,
                        help="Angle resolution for azimuth")

    parser_args = parser.parse_args()

    df['labels'] = (df['azimuth'] + 90)
    if parser_args.azimuth_resolution:
        df['labels'] = df['labels'] // parser_args.azimuth_resolution
    main(df)
