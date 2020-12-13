import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler


def get_preprocess(n_inputs):
    def preprocess(line):
        defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
        fields = tf.io.decode_csv(line, record_defaults=defs)
        x = tf.stack(fields[:-1])
        y = tf.stack(fields[-1:])
        return x, y / 10000
    return preprocess


def get_preprocess_norm(n_inputs, X_mean, X_std):
    def preprocess(line):
        defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
        fields = tf.io.decode_csv(line, record_defaults=defs)
        x = tf.stack(fields[:-1])
        y = tf.stack(fields[-1:])
        return (x - X_mean) / X_std, y
    return preprocess


def get_mean_std(data):
    scaler = StandardScaler()
    scaler.fit(data)
    X_mean = scaler.mean_
    X_std = scaler.scale_

    return X_mean, X_std


def csv_reader_dataset(filepaths, dir, preprocess_fn, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(dir + filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


def get_callbacks(callback_name):
    all_callbacks = []
    if "save_best_only" in callback_name:
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./saved_model/NN_best.h5", save_best_only=True)
        all_callbacks.append(checkpoint_cb)
    if "EarlyStopping" in callback_name:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        all_callbacks.append(early_stopping_cb)

    return all_callbacks

def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="selu", input_shape=[2]),
        keras.layers.Dense(5, activation="selu"),
        keras.layers.Dense(1, activation="relu")
    ])

    return model


def main():
    Train_dir = "../data/data_2in1out/"
    Valid_dir = "../data/data_2in1out/valid/"
    train_filepaths = "PV*.csv"
    valid_filepaths = "PV*.csv"

    n_inputs = 2  # 2 inputs: time, temperature
    batch_size = 16
    preprocess_fn = get_preprocess(n_inputs)
    train_set = csv_reader_dataset(train_filepaths, Train_dir, preprocess_fn,
                                   repeat=None, batch_size=batch_size)  # Put Training set into pipeline
    valid_set = csv_reader_dataset(valid_filepaths, Valid_dir, preprocess_fn)

    #for X_batch, y_batch in train_set.take(3):
    #    print(X_batch)
    #    print(y_batch)

    model = create_model()
    early_stopping = get_callbacks(["save_best_only"])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(train_set, steps_per_epoch=6000 // batch_size, epochs=50, validation_data=valid_set,
              callbacks=early_stopping)
    y_pred = model.predict(x=[[1, 10]])
    print(y_pred)


if __name__ == '__main__':
    main()
