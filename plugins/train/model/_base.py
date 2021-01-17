import logging
import sys
import os
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.models import load_model

from lib.utils import FaceswapError

logger = logging.getLogger(__name__)


class ModelBase():
    """ Base class that all model plugins should inherit from """
    def __init__(self, model_dir, arguments, predict=False):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', arguments: %s, predict: %s)",
                      self.__class__.__name__, model_dir, arguments, predict)
        self.input_shape = None
        self._args = arguments
        self._is_predict = predict
        self._model = None
        self._io = _IO(self, model_dir, self._is_predict)
        self._check_multiple_models()

    @property
    def name(self):
        """ str: The name of this model based on the plugin name. """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        return os.path.splitext(basename)[0].lower()

    @property
    def model(self):
        return self._model

    @property
    def model_dir(self):
        """ str: The full path to the model folder location """
        return self._io._model_dir

    def _check_multiple_models(self):
        multiple_models = self._io.multiple_models_in_folder
        if multiple_models is None:
            logger.debug("Contents of floder are valid")
            return
        if len(multiple_models) == 1:
            msg = ("You have requested to train with the '{}' plugin, but a model file for the "
                   "'{}' plugin already exists in the folder '{}'.\nPlease select a different "
                   "model folder.".format(self.name, multiple_models[0], self.model_dir))
        else:
            msg = ("There are multiple plugin types ('{}') stored in the model folder '{}'. This "
                   "is not supported.\nPlease split the model files into their own folders before "
                   "proceeding".format("', '".join(multiple_models), self.model_dir))
        raise FaceswapError(msg)

    def build(self):
        if self._io.model_exists:
            model = self._io._load()
            if self._is_predict:
                # currently don't need to do anything
                self._model = model
                pass
            else:
                self._model = model
        else:
            self._model = self.build_model(self.input_shape)

        if not self._is_predict:
            self._compile_model()
        self._out_summary()

    def build_model(self, inputs):
        """ Override for Model Specific autoencoder builds """
        raise NotImplementedError

    def _out_summary(self):
        self._model.summary(print_fn=lambda x: logger.verbose("%s", x))

    def _compile_model(self):
        """ Compile the model to include the Optimizer and Loss Function(s) """
        # TODO: add config file
        optimizer = Adam(learning_rate=0.01)
        self._model.compile(optimizer=optimizer, loss="mse")


class _IO():
    def __init__(self, plugin, model_dir, is_predict):
        self._plugin = plugin
        self._model_dir = model_dir
        self._is_predict = is_predict

    @property
    def _filename(self):
        """str: The filename for this model."""
        return os.path.join(self._model_dir, "{}.h5".format(self._plugin.name))  # name is a property function of _base

    @property
    def model_exists(self):
        return os.path.isfile(self._filename)

    @property
    def multiple_models_in_folder(self):
        plugins = [fname.replace(".h5", "") for fname in os.listdir(self._model_dir) if fname.endswith(".h5")]

        # return None if nothing in the folder or model name matches with plugin name
        test_names = plugins + [self._plugin.name]
        test = False if not test_names else os.path.commonprefix(test_names) == ""
        retval = None if not test else plugins
        return retval

    def _load(self):
        if self._is_predict and not self.model_exists:
            logger.error("Model could not be found in folder '%s'. Exiting", self._model_dir)
            sys.exit(1)

        model = load_model(self._filename, compile=False)
        return model
