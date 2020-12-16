"""
Sequence to vector model, given input sequence, only predict the next time step
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from prepare_data import get_data_unscaled_pure_vec


def create_model(seq_length):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[seq_length, 1]),
        keras.layers.Dense(1)
    ])
    return model


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
    seq_length = 10
    interval = 10
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_data_unscaled_pure_vec(data_filename, seq_length, interval)
    print(X_train.shape)
    print(Y_train.shape)
    # print(X_train[2])
    # print(Y_train[2])
    model = create_model(seq_length)
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.01))

    early_stopping = get_callbacks(["save_best_only"])
    exponential_decay_fn = exponential_decay(lr0=0.01, s=300)  # 10 times smaller after s epochs
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)  # Automatically carry lr0 to next epoch
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_valid, Y_valid),
                        callbacks=[lr_scheduler, early_stopping])
    y_pred = model.predict(X_train[:1])
    print("y_pred: ", y_pred)


if __name__ == '__main__':
    main()
