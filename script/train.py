import os
import logging
from plugins.plugin_loader import PluginLoader


class Train():
    def __init__(self, arguments):
        logging.warning("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._data_dir = self._args.data_dir
        self._my_arg2 = self._args.test_arg2
        logging.warning("Test arg1: %s, arg2 %s", self._my_arg1, self._my_arg2)

    def process(self):
        """ The entry point for Training process """
        logging.warning("processing...")
        self._run_training()

    def _run_training(self):
        """ The main Trainign process """
        logging.warning("Start traning... args1: %s", self._my_arg1)
        model = self._load_model()
        trainer = self._load_trainer(model)
        self._run_training_cycle(model, trainer)

    def _load_model(self):
        logging.debug("Loading Model...")
        model = PluginLoader.get_model("naive_NN")(
            model_dir=None,
            arguments=None
        )   # TODO: create a name variable
        model.build()
        return model

    def _load_trainer(self, model):
        trainer = PluginLoader.get_trainer(model.trainer)
        trainer = trainer(model, self._data_dir, self._args.batch_size)
        return trainer

    def _run_training_cycle(self, model, trainer):
        for iteration in range(1, self._args.iterations + 1):
            trainer.train_one_step()
        model.save()

