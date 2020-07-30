import pandas as pd
from ml.CONFIG import *
from models import *
from dataGenerator import DataGenerator
import argparse



def get_datasets(df_input, val=False):
    df_test = df_input[df_input['subject_id'].isin(TEST_SUBJECTS)]
    df_train = df_input.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if val:
        df_val = df_train.sample(frac=0.1, random_state=random_state)
        df_train = df_train.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        return df_train, df_val, df_train

    return df_train, df_test

def main(df_input):
    output_shape = int(df_input['labels'].max() + 1)

    # Create a new model instance
    model = get_model_cnn(output_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

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

    test_generator = DataGenerator(df_test, FEATURE, output_shape, **params)

    # Re-evaluate the model
    los, acc = model.evaluate(test_generator, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return 1


if __name__ == "__main__":
    df = pd.read_csv(PATH_DATASET)
    parser = argparse.ArgumentParser()
    parser.add_argument("--azimuth_resolution", type=int, default=-1,
                        help="Angle resolution for azimuth")

    parser_args = parser.parse_args()

    if parser_args.azimuth_resolution:
        df['labels'] = (df['azimuth'] + abs(df['azimuth'].min()))
        df['labels'] = df['labels'] // parser_args.azimuth_resolution
    main(df)
