import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import get_data_unscaled_pure_vec_shaffle


def last_time_step_mse(Y_true, Y_pred):
    """ For validation, only care the error at the last step """
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def main():
    data_filename = "../data/data_seq2seq/data_2features/raw_data.csv"
    seq_length = 50
    interval = 10
    _, _, X_test, _, _, Y_test = get_data_unscaled_pure_vec_shaffle(data_filename, seq_length, interval)
    print(X_test.shape)

    model = keras.models.load_model("./saved_model/seq2vec_Deep_RNN.h5")
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    Y_true = Y_test

    index = 18
    plt.plot(np.arange(0, seq_length), X_test[index, :, 0])
    plt.plot(seq_length, Y_true[index], 'b*')
    plt.plot(seq_length, y_pred[index], 'rx')
    plt.show()

    mse = ((Y_true - y_pred) ** 2).mean(axis=None)
    mse2 = keras.metrics.mean_squared_error(Y_true, y_pred)
    print("mse: ", mse)
    print("mse2: ", tf.reduce_sum(mse2) / len(Y_true))

if __name__ == '__main__':
    main()
