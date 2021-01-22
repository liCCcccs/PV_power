import os
import logging
from plugins.plugin_loader import PluginLoader


class Train():
    def __init__(self, arguments):
        logging.info("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._data_dir = self._args.data_dir
        self._epochs = self._args.epochs
        self._model_name = self._args.model
        self._location = self._args.location
        self._all_feature = self._args.all_feature
        self._feature_name = self._args.feature_name
        logging.warning("Test arg1: %s, arg2 %s", self._data_dir, self._epochs)

    def process(self):
        """ The entry point for Training process """
        logging.info("processing...")
        self._run_training()

    def _run_training(self):
        """ The main Trainign process """
        logging.info("Start traning... args1: %s", self._data_dir)
        model = self._load_model()
        trainer = self._load_trainer(model)
        self._run_training_cycle(model, trainer)

    def _load_model(self):
        logging.info("Loading Model...")
        model = PluginLoader.get_model(self._model_name)(
            model_dir=None,
            arguments=None
        )
        model.build()
        return model

    def _load_trainer(self, model):
        trainer = PluginLoader.get_trainer(model.trainer)
        if self._model_name == "NN":
            use_pipeline = False
        else:
            use_pipeline = True
        if self._feature_name == "date":
            feature_index = 0
        elif self._feature_name == "time":
            feature_index = 1
        else:
            feature_dict = self._load_feature_name()
            feature_index = feature_dict[self._feature_name] + 2
        trainer = trainer(model, self._data_dir, self._args.batch_size, use_pipeline, feature_index)
        return trainer

    def _load_feature_name(self):
        inspector = PluginLoader.get_inspector(self._location)()
        feature_name = inspector.get_feature_name()
        return feature_name

    def _run_training_cycle(self, model, trainer):
        trainer.train_cycle(epochs=self._epochs)
        model.model().save("./my_saved_model/test1_naive_NN.h5")
        logging.info("model has been saved ============")

