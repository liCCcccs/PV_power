import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler


def prepare_data(filename):
    raw_data = pd.read_csv(filename, sep=',', header=0).values  # read to a list
    raw_data = np.array(raw_data)  # leave the last 10 data points for y
    n_steps_ahead = 10
    seq_length = 100
    n_steps = seq_length + n_steps_ahead   # length of a piece of signal
    num_batch = raw_data.shape[0] // n_steps
    batched_data = np.reshape(raw_data[:n_steps*num_batch], [num_batch, n_steps, 3])  # shape [80, 110, 3]

    X_train = batched_data[:65, :seq_length, :]
    X_valid = batched_data[65:75, :seq_length, :]
    X_test = batched_data[75:, :seq_length, :]

    Y = np.empty((num_batch, seq_length, n_steps_ahead))
    for step_ahead in range(1, n_steps_ahead + 1):
        Y[..., step_ahead - 1] = batched_data[..., step_ahead:step_ahead + seq_length, 2]
    Y_train = Y[:65] / 10000
    Y_valid = Y[65:75] / 10000
    Y_test = Y[75:] / 10000

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def create_model():
    model = keras.models.Sequential([
        keras.layers.GRU(20, return_sequences=True, input_shape=[None, 3]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10, activation="relu"))
    ])
    return model


def last_time_step_mse(Y_true, Y_pred):
    """ For validation, only care the error at the last step """
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def get_callbacks(callback_name):
    all_callbacks = []
    if "save_best_only" in callback_name:
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./saved_model/seq2seq_2in_GRU.h5", save_best_only=True)
        all_callbacks.append(checkpoint_cb)
    if "EarlyStopping" in callback_name:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        all_callbacks.append(early_stopping_cb)

    return all_callbacks


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


def main():
    data_filename = "../data/data_seq2seq/data_2features/raw_data.csv"
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prepare_data(data_filename)
    print(X_train.shape)
    print(Y_train.shape)
    model = create_model()
    model.compile(loss="mse", optimizer=keras.optimizers.Nadam(lr=0.001), metrics=[last_time_step_mse])

    early_stopping = get_callbacks(["save_best_only"])
    exponential_decay_fn = exponential_decay(lr0=0.001, s=300)  # 10 times smaller after s epochs
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)  # Automatically carry lr0 to next epoch
    history = model.fit(X_train, Y_train, epochs=300, validation_data=(X_valid, Y_valid),
                        callbacks=[lr_scheduler, early_stopping])


if __name__ == '__main__':
    main()
