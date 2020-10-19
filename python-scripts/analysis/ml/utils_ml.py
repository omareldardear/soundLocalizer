
#######################################################################
# FUNCTIONS                                                           #
#######################################################################
import tensorflow as tf

random_state = 42



def focal_loss(gamma=1., alpha=0.5):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.compat.v1.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


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
