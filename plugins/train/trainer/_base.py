import logging
from lib.training_data import TrainingDataGenerator


class TrainerBase():
    def __init__(self, model, data_dir, batch_size, use_pipeline, feature_index):
        logging.warning("Initializing trainer...")
        self._model = model
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._feature_index = feature_index
        self._feeder = self._Feeder(self._data_dir,
                                    self._model,
                                    self._feature_index,
                                    self._batch_size,
                                    use_pipeline=use_pipeline)  # return a tensorflow Dataset

    def train_cycle(self, epochs):
        train_set = self._feeder.get_dataset()
        history = self._model.model.fit(train_set, epochs=epochs, steps_per_epoch=100)  # TODO: check this 100

    class _Feeder():
        def __init__(self, data_dir, model, feature_index, batch_size, use_pipeline):
            self._model = model
            self._data_dir = data_dir
            self._feeds = self._load_generator().generate_dataset(data_dir, feature_index, batch_size, use_pipeline)

        def _load_generator(self):
            input_size = self._model.model.input_shape[1]  # model.input_shape = (None, n)
            generator = TrainingDataGenerator(input_size)
            return generator

        def get_dataset(self):
            return self._feeds

