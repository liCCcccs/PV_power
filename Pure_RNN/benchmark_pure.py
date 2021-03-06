"""
Predict the next 1 points to be the same as the last one. Use this error as the benchmark
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import get_data_unscaled_pure_vec


def main():
    data_filename = "../data/data_seq2seq/data_2features/raw_data.csv"
    seq_length = 10
    interval = 10
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_data_unscaled_pure_vec(data_filename, seq_length, interval)
    print(X_train.shape)
    print(Y_train.shape)

    X_keep = np.zeros([X_train.shape[0], 1])
    for i in range(X_train.shape[0]):
        X_keep[i, 0] = X_train[i, -1, 0]

    X_true = Y_train
    print(X_keep.shape)
    print(X_true.shape)

    index = 12
    plt.plot(np.arange(0, seq_length), X_train[index, :, 0])
    plt.plot(seq_length, X_true[index], 'b*')
    plt.plot(seq_length, X_keep[index], 'rx')
    plt.show()

    mse = ((X_true - X_keep) ** 2).mean(axis=None)
    mse2 = keras.metrics.mean_squared_error(X_true, X_keep)
    print("mse: ", mse)
    print("mse2: ", tf.reduce_sum(mse2) / len(X_true))



if __name__ == '__main__':
    main()
