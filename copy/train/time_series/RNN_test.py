"""
In this script, RNN forecast multiple steps ahead
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


Data_path = "../../Yulara/read_to_train/StandardScaler/"
Train_all = True
Feature_name = "time"
Feature_dict = {"date": 0, "time": 1, "wind_speed": 2, "temperature": 3, "radiation": 4, "wind_direction": 5, "rainfall": 6,
                         "max_wind_speed": 7, "air_pressure": 8, "hail_accumulation": 9, "pyranometer_1": 10,
                         "temperature_probe_1": 11, "temperature_probe_2": 12, "AEDR": 13, "Active_Power": 14}

K = keras.backend


def csv_reader_numpy(mypath):
    """ Read all csv files into a numpy array """
    file = "../../Yulara/read_to_train/StandardScaler/Yulara_2018.csv"
    data = np.loadtxt(file, delimiter=",", dtype=float)
    return data


def diff_data(all_data, n_steps):
    all_data = all_data[8000:9202]
    X, y = all_data[:, :-1], all_data[:, -1]
    y /= 1000  # scale down
    print(X.shape, y.shape)
    X = np.c_[(X[1:, :], y[:-1])]  # e.g. X = [whether 8pm, power 7pm], y = [power 8pm]
    y = y[1:]  # remove the first row
    print("=====================")
    print(X.shape, y.shape)
    num_pred = 9202 - 8000 - n_steps
    X_feed = np.zeros((num_pred, n_steps, 15))
    for i in range(num_pred):
        X_feed[i] = X[i:i + n_steps]

    print(X_feed.shape)
    return X_feed, y[n_steps - 1:]


def split(X, y):
    X_train_all, X_test, y_train_all, y_test = X[:8000], X[8000:], y[:8000], y[8000:]
    X_train, X_valid, y_train, y_valid = X_train_all[:7000], X_train_all[7000:], y_train_all[:7000], y_train_all[7000:]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def forecast_option1(X_feed, y_true, model):
    """ Using model trained for forecasting one step ahead, iteratively forecast multiple steps """
    y_pred = model.predict(X_feed)

    print(y_pred.shape, y_true.shape)
    y_pred = y_pred[:, 0]  # remove redundant dimension
    y_pred = y_pred.clip(min=0)   # restrict power to be non-negative
    y_pred *= 100 * 0.8
    y_true *= 100 * 0.8
    mse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print("mse: ", mse)

    plt.plot(y_true)
    plt.plot(y_pred)
    plt.savefig("pred.png")


def main():
    n_steps = 20
    all_data = csv_reader_numpy(Data_path)
    X_feed, y_true = diff_data(all_data, n_steps)

    model = keras.models.load_model("./saved_model/LSTM_all.h5")
    forecast_option1(X_feed, y_true, model)


if __name__ == "__main__":
    main()