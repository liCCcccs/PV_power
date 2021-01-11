import tensorflow as tf
from tensorflow import keras
from ._base import ModelBase
import logging


class Model(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = [2]
        self.trainer = "pv_trainer"

    def build_model(self, inputs):
        logging.warning("building model... input_shape: %s", inputs)
        model = keras.models.Sequential([
            keras.layers.Dense(10, activation="selu", input_shape=inputs),
            keras.layers.Dense(5, activation="selu"),
            keras.layers.Dense(1, activation="relu")
        ])
        return model
