"""
Predict the next 10 points to be the same as the last one. Use this error as the benchmark
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import get_data_unscaled


def main():
    data_filename = "../data/data_seq2seq/data_2features/raw_data.csv"
    n_steps_ahead = 10
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_data_unscaled(data_filename, n_steps_ahead)
    print(X_train.shape)
    print(Y_train.shape)

    X_keep = np.zeros([X_train.shape[0], n_steps_ahead])
    for i in range(X_train.shape[0]):
        X_keep[i, :] = np.full(n_steps_ahead, X_train[i, -1, 2])

    X_true = Y_train[:, -10:, -1]
    print(X_keep.shape)
    print(X_true.shape)

    plt.plot(np.arange(0, 100), X_train[6, :, 2])
    plt.plot(np.arange(100, 110), X_true[6, :])
    plt.plot(np.arange(100, 110), X_keep[6, :])
    #plt.show()

    mse = ((X_true - X_keep) ** 2).mean(axis=None)
    mse2 = keras.metrics.mean_squared_error(X_true, X_keep)
    print("mse: ", mse)
    print("mse2: ", tf.reduce_sum(mse2) / 4)



if __name__ == '__main__':
    main()
