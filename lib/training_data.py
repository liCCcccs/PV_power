import tensorflow as tf


class TrainingDataGenerator():
    def __init__(self, model_input_size):
        self._model_input_size = model_input_size

    def generate_dataset(self, data_dir, batch_size):
        file_pattern = "PV*.csv"
        preprocess_fn = self._get_preprocess(self._model_input_size)
        train_set = self._csv_reader_dataset(file_pattern, data_dir, preprocess_fn,
                                             repeat=None, batch_size=batch_size)  # Put Training set into pipeline
        return train_set

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
