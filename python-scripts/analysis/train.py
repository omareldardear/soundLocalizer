import pandas as pd
from utils import *
from CONFIG import *
from models import *
import tensorflow as tf
from dataGenerator import DataGenerator
random_state = 42



def get_datasets(df_input, val=False):
    df_test = df_input[df_input['subject_id'].isin(TEST_SUBJECTS)]
    df_train = df_input.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if val:
        df_val = df_train.sample(frac=0.2, random_state=random_state)
        df_train = df_train.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        return df_train, df_val, df_train

    return df_train, df_test

###################################################################################
#                                   MAIN PROCESS                                  #
###################################################################################


def main(df_input):
    output_shape = int(df_input['labels'].max() + 1)

    features_to_normalize = ['joint2', 'joint0']
    df[features_to_normalize] = df_input[features_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Model parameters
    params = {'dim': INPUT_SHAPE,
              'batch_size': BATCH_SIZE,
              'n_channels': 1,
              'resample': RESAMPLING_F,
              'shuffle': True}

    # Define train and test generators
    df_train, df_val, df_test = get_datasets(df_input, True)

    training_generator = DataGenerator(df_train, FEATURE, output_shape, **params)
    val_generator = DataGenerator(df_val, FEATURE, output_shape, **params)
    test_generator = DataGenerator(df_test, FEATURE, output_shape, **params)

    # Define the model
    model = get_model_cnn(output_shape)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INIT_LR,
        decay_steps=500,
        decay_rate=0.94,
        name="lr_decay"
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(training_generator, callbacks=get_callbacks(), epochs=EPOCHS, validation_data=val_generator)

    # Re-evaluate the model
    los, acc = model.evaluate(test_generator, verbose=2)
    print("Test model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('/tmp/data/saved_model/my_model.h5')


if __name__ == '__main__':
    df = pd.read_csv(PATH_DATASET)
    df = df.sample(frac=1).reset_index(drop=True)
    main(df)
