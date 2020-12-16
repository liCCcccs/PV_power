import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import get_data_unscaled_pure


def last_time_step_mse(Y_true, Y_pred):
    """ For validation, only care the error at the last step """
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def main():
    data_filename = "../data/data_seq2seq/data_2features/raw_data.csv"
    n_steps_ahead = 1
    X_train, _, X_test, Y_train, _, Y_test = get_data_unscaled_pure(data_filename, n_steps_ahead)
    print(X_train.shape)

    model = keras.models.load_model("./saved_model/seq2seq_2in_RNN.h5",
                                    custom_objects={"last_time_step_mse": last_time_step_mse})
    test_index = 3
    y_pred = model.predict(X_train)
    print(y_pred.shape)

    #plt.plot(np.arange(100), X_test[test_index, :, 2] / 10000)
    #plt.plot(np.arange(100, 110), y_pred[0, -1, :])
    #plt.plot(np.arange(100, 110), Y_test[test_index, -1, :])
    #plt.show()
    y_pred_last = y_pred[:, -1, :]
    y_true_last = Y_train[:, -1, :]
    print(y_pred_last.shape)
    print(y_true_last.shape)
    mse = ((y_pred_last - y_true_last) ** 2).mean(axis=None)
    print("mse: ", mse)


if __name__ == '__main__':
    main()
