import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from prepare_data import get_data_unscaled_pure


def create_model():
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10, activation="relu"))
    ])
    return model


def last_time_step_mse(Y_true, Y_pred):
    """ For validation, only care the error at the last step """
    print(Y_true.shape, Y_pred.shape)
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def get_callbacks(callback_name):
    all_callbacks = []
    if "save_best_only" in callback_name:
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./saved_model/seq2seq_2in_RNN.h5", save_best_only=True)
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
    n_steps_ahead = 1
    seq_length = 100
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_data_unscaled_pure(data_filename, n_steps_ahead, seq_length)
    print(X_train.shape)
    print(Y_train.shape)
    model = create_model()
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.0005), metrics=[last_time_step_mse])

    early_stopping = get_callbacks(["save_best_only"])
    exponential_decay_fn = exponential_decay(lr0=0.0005, s=300)  # 10 times smaller after s epochs
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)  # Automatically carry lr0 to next epoch
    history = model.fit(X_train, Y_train, batch_size=32, epochs=300, validation_data=(X_valid, Y_valid),
                        callbacks=[lr_scheduler, early_stopping])


if __name__ == '__main__':
    main()
