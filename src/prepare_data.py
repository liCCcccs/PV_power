import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_data_unscaled(filename, n_steps_ahead):
    raw_data = pd.read_csv(filename, sep=',', header=0).values  # read to a list
    raw_data = np.array(raw_data)  # leave the last 10 data points for y
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
