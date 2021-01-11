import logging


class TrainerBase():
    def __init__(self, model, data_dir, batch_size):
        logging.warning("Initializing trainer...")
        self._model = model
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._feeder = self._Feeder(self._data_dir,
                     self._model)  # return a tensorflow Dataset

    def train_one_step(self):
        # TODO: implement tf.Dataset next() in get_batch()
        model_inputs, model_targets = self._feeder.get_batch()
        loss = self._model.model.train_on_batch(model_inputs, y=model_targets)

    class _Feeder():
        def __init__(self, data_dir, model):
            self._model = model
            self._data_dir = data_dir
            self._feeds = self._load_generator()

        def _load_generator(self):
            input_size = self._model.model.input_shape[0]
            generator = TrainingDataGenerator(input_size)

        def get_batch(self):
            pass

