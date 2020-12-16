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


def get_data_unscaled_pure(filename, n_steps_ahead, seq_length):
    raw_data = pd.read_csv(filename, sep=',', header=0).values  # read to a list
    raw_data = np.array(raw_data)  # leave the last 10 data points for y
    n_steps = seq_length + n_steps_ahead   # length of a piece of signal
    num_batch = raw_data.shape[0] // n_steps
    batched_data = np.reshape(raw_data[:n_steps*num_batch], [num_batch, n_steps, 3])  # shape [80, 110, 3]

    X_train = batched_data[:65, :seq_length, 2:]
    X_valid = batched_data[65:75, :seq_length, 2:]
    X_test = batched_data[75:, :seq_length, 2:]

    Y = np.empty((num_batch, seq_length, n_steps_ahead))
    for step_ahead in range(1, n_steps_ahead + 1):
        Y[..., step_ahead - 1] = batched_data[..., step_ahead:step_ahead + seq_length, 2]
    Y_train = Y[:65] / 10000
    Y_valid = Y[65:75] / 10000
    Y_test = Y[75:] / 10000

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def get_data_unscaled_pure_vec(filename, seq_length, interval):
    """ For seq2vec model, only predict one step ahead """
    raw_data = pd.read_csv(filename, sep=',', header=None).values  # read to a list
    raw_data = np.array(raw_data)  # leave the last 10 data points for y
    X_train_raw = raw_data[:6500]
    X_valid_raw = raw_data[6500:8000]
    X_test_raw = raw_data[8000:]
    num_batch_train = (X_train_raw.shape[0] - seq_length - 1) // interval  # include the one predict step
    num_batch_valid = (X_valid_raw.shape[0] - seq_length - 1) // interval
    num_batch_test = (X_test_raw.shape[0] - seq_length - 1) // interval

    X_train = np.zeros([num_batch_train, seq_length, 1])
    X_valid = np.zeros([num_batch_valid, seq_length, 1])
    X_test = np.zeros([num_batch_test, seq_length, 1])
    for batch in range(num_batch_train):
        X_train[batch, :, 0] = X_train_raw[batch*interval:batch*interval + seq_length, 2]
    for batch in range(num_batch_valid):
        X_valid[batch, :, 0] = X_valid_raw[batch*interval:batch*interval + seq_length, 2]
    for batch in range(num_batch_test):
        X_test[batch, :, 0] = X_test_raw[batch*interval:batch*interval + seq_length, 2]

    Y_train = np.zeros([num_batch_train, 1])
    Y_valid = np.zeros([num_batch_valid, 1])
    Y_test = np.zeros([num_batch_test, 1])
    for batch in range(num_batch_train):
        Y_train[batch] = X_train_raw[batch*interval + seq_length: batch*interval + seq_length + 1, 2]
    for batch in range(num_batch_valid):
        Y_valid[batch] = X_valid_raw[batch*interval + seq_length: batch*interval + seq_length + 1, 2]
    for batch in range(num_batch_test):
        Y_test[batch] = X_test_raw[batch*interval + seq_length: batch*interval + seq_length + 1, 2]
    X_train /= 10000
    X_valid /= 10000
    X_test /= 10000
    Y_train /= 10000
    Y_valid /= 10000
    Y_test /= 10000

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
