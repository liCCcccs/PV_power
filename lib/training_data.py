import os
from os import listdir
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from plugins.plugin_loader import PluginLoader


class TrainingDataGenerator():
    def __init__(self, model_input_size):
        self._model_input_size = model_input_size

    def _get_file_pattern(self, data_dir):
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]
        location = os.path.split(data_dir)[-1]
        return location + "_*.csv"  # e.g. "Yulara_*.csv"

    def generate_dataset(self, data_dir, feature_index, std_method, batch_size, use_pipeline):
        file_pattern = self._get_file_pattern(data_dir)
        if use_pipeline:
            preprocess_fn = self._get_preprocess(self._model_input_size)
            train_set = self._csv_reader_dataset(file_pattern, data_dir, preprocess_fn,
                                                 repeat=None, batch_size=batch_size)  # Put Training set into pipeline
            return train_set
        else:
            all_data = self._csv_reader_numpy(os.path.join(data_dir, std_method))  # e.g. "./data/Yulara/MinMAxScaler"
            all_x, all_y = all_data[:, :-1], all_data[:, -1:]
            X_train_all, X_test, y_train_all, y_test = train_test_split(all_x, all_y, test_size=0.05, random_state=42)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all)
            return X_train, X_valid, X_test, y_train, y_valid, y_test

    def _csv_reader_numpy(self, mypath):
        """ Read all csv files into a numpy array """
        onlyfiles = [f for f in listdir(mypath) if f.endswith(".csv")]
        all_data = None
        for index, file in enumerate(onlyfiles):
            data = np.loadtxt(file, delimiter=",", dtype=float)
            if index == 0:
                all_data = data
            else:
                all_data = np.vstack((all_data, data))
        return all_data

    def _get_preprocess(self, n_inputs):
        def preprocess(line):
            defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
            fields = tf.io.decode_csv(line, record_defaults=defs)
            x = tf.stack(fields[:-1])
            y = tf.stack(fields[-1:])
            return x, y / 10000
        return preprocess

    def _csv_reader_dataset(self, file_pattern, data_dir, preprocess_fn, repeat=1, n_readers=5,
                            n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):
        """ Read all the .csv files in data_dir and put them in a tf dataset """
        dataset = tf.data.Dataset.list_files(data_dir + file_pattern).repeat(repeat)
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
            cycle_length=n_readers, num_parallel_calls=n_read_threads)

        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(preprocess_fn, num_parallel_calls=n_parse_threads)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)
