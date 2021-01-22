import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


Data_path = "../../Yulara/read_to_train/MinMaxScaler/"
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


def split_data(all_data):
    if Train_all == True:
        all_x, all_y = all_data[:, :-1], all_data[:, -1:]
    else:
        feature_col = Feature_dict[Feature_name]
        all_x, all_y = all_data[:, feature_col:feature_col + 1], all_data[:, -1:]

    all_y /= 1000   # normalize power
    X_train_all, X_test, y_train_all, y_test = train_test_split(all_x, all_y, test_size=0.05, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.1)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_model(input_size):
    model = keras.models.Sequential([
        keras.layers.Dense(36, activation="relu", input_shape=[input_size]),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])
    return model


def  build_model_all(input_size):
    model = keras.models.Sequential([
        keras.layers.Dense(40, activation="relu", input_shape=[input_size]),
        keras.layers.Dense(80, activation="relu"),
        keras.layers.Dense(40, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])
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
    all_data = csv_reader_numpy(Data_path)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(all_data)
    if Train_all:
        input_size = len(Feature_dict) - 1
        save_name = "./saved_model/NN_all.h5"
    else:
        input_size = 1
        save_name = "./saved_model/NN_" + Feature_name + ".h5"

    if Train_all:
        model = build_model_all(input_size)
    else:
        model= build_model(input_size)

    early_stopping = get_callbacks(["save_best_only"], save_name)
    exponential_decay_fn = exponential_decay(lr0=0.2, s=20)
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-1))
    model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid),
              callbacks=[early_stopping, lr_scheduler])
    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()
