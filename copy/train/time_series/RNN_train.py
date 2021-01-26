import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


Data_path = "../../Yulara/read_to_train/StandardScaler_2018/"
Train_all = True
Feature_name = "time"
Feature_dict = {"date": 0, "time": 1, "wind_speed": 2, "temperature": 3, "radiation": 4, "wind_direction": 5, "rainfall": 6,
                         "max_wind_speed": 7, "air_pressure": 8, "hail_accumulation": 9, "pyranometer_1": 10,
                         "temperature_probe_1": 11, "temperature_probe_2": 12, "AEDR": 13, "Active_Power": 14}

K = keras.backend


def csv_reader_numpy(mypath):
    """ Read all csv files into a numpy array """
    onlyfiles = [f for f in os.listdir(mypath) if f.endswith(".csv")]
    all_data = None
    for index, file in enumerate(onlyfiles):
        data = np.loadtxt(Data_path + file, delimiter=",", dtype=float)
        if index == 0:
            all_data = data
        else:
            all_data = np.vstack((all_data, data))
    return all_data


def diff_data(all_data, n_steps):
    X, y = all_data[:, :-1], all_data[:, -1]
    y /= 1000   # scale down
    print(X.shape, y.shape)
    X = np.c_[(X[1:, :], y[:-1])]    # e.g. X = [whether 8pm, power 7pm], y = [power 8pm]
    y = y[1:]   # remove the first row
    print("=====================")
    print(X.shape, y.shape)
    res = len(X) % n_steps
    n_features = X.shape[-1]
    shaped_X = np.reshape(X[:-res], (-1, n_steps, n_features))
    print(shaped_X[-1].shape)
    shaped_y = y[n_steps::n_steps]

    print(shaped_X.shape, shaped_y.shape)
    return shaped_X, shaped_y

def split(X, y, n_steps):
    total = X.shape[0]
    test_num = total - int(1300 / n_steps)
    valid_num = test_num - 200
    X_train_all, X_test, y_train_all, y_test = X[:test_num], X[test_num:], y[:test_num], y[test_num:]
    X_train, X_valid, y_train, y_valid = X_train_all[:valid_num], X_train_all[valid_num:], y_train_all[:valid_num], y_train_all[valid_num:]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model(n_steps):
    model = keras.models.Sequential([
        keras.layers.GRU(20, input_shape=[None, 15]),
        keras.layers.Dense(1)
    ])
    model.summary()
    return model


def get_callbacks(callback_name, save_name):
    all_callbacks = []
    if "save_best_only" in callback_name:
        checkpoint_cb = keras.callbacks.ModelCheckpoint(save_name, save_best_only=True)
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
    n_steps = 20
    all_data = csv_reader_numpy(Data_path)
    shaped_X, shaped_y = diff_data(all_data, n_steps)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split(shaped_X, shaped_y, n_steps)

    model = create_model(n_steps)

    save_name = "./saved_model/LSTM_all.h5"
    early_stopping = get_callbacks(["save_best_only"], save_name)
    exponential_decay_fn = exponential_decay(lr0=0.001, s=10)
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

    model.compile(loss="mse", optimizer=keras.optimizers.Nadam(lr=0.001))
    history = model.fit(X_train, y_train, batch_size=8, epochs=20, validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping, lr_scheduler])
    model.evaluate(X_test, y_test)
    model.save("./saved_model/LSTM_all.h5")


if __name__ == "__main__":
    main()
