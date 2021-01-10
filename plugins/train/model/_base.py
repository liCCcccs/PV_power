import logging


class ModelBase():
    """ Base class that all model plugins should inherit from """
    def __init__(self, model_dir, arguments, predict=False):
        logging.debug("Initializing ModelBase (%s): (model_dir: '%s', arguments: %s, predict: %s)",
                      self.__class__.__name__, model_dir, arguments, predict)
        self.input_shape = None
        self._args = arguments
        self._is_predict = predict
        self._model = None

    def model(self):
        return self._model

    def build(self):
        if self._is_predict:
            # load exsiting model and predit
            pass
        else:
            self._model = self.build_model(self.input_shape)

    def build_model(self, inputs):
        """ Override for Model Specific autoencoder builds """
        raise NotImplementedError
