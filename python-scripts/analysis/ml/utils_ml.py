
#######################################################################
# FUNCTIONS                                                           #
#######################################################################
import tensorflow as tf

random_state = 42

def get_datasets(df_input, test_subjects, val=False):
    df_test = df_input[df_input['subject_id'].isin(test_subjects)]
    df_train = df_input.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    if val:
        df_val = df_train.sample(frac=0.1, random_state=random_state)
        df_train = df_train.drop(df_val.index).reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        return df_train, df_val, df_test

    return df_train, df_test



def get_callbacks(m_patience=20):
    """
    Get callback function for the training
    :return: Earlystopping, Tensorboard-log, Saving model
    """
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "/tmp/training_2/cp-{epoch:04d}.ckpt"

    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=m_patience),
        tf.keras.callbacks.TensorBoard("data/log"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=10)
    ]
