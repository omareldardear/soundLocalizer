import pandas as pd
from utils import *
from CONFIG import *
from models import *



def main(df):
    output_shape = df['labels'].max() + 1

    # Create a new model instance
    model = get_model_dense_simple(output_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])


    # Load the previously saved weights
    latest = '/tmp/training_2/cp-0020.ckpt'
    model.load_weights(latest)

    df_test = df[df['subject_id'].isin(TEST_SUBJECTS)]
    labels = tf.keras.utils.to_categorical(df['labels'])



    ds_test = tf.data.Dataset.from_generator(
        lambda: sound_location_generator(df_test, labels),
        (tf.float32, tf.int64), ((None, 1), output_shape)
    ).batch(BATCH_SIZE)

    # Re-evaluate the model
    los, acc = model.evaluate(ds_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('data/saved_model/my_model')

    return 1


if __name__ == "__main__":
    df = pd.read_csv(PATH_DATASET)

    main(df)
