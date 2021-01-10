import tensorflow as tf
from tensorflow import keras
from ._base import ModelBase


class Model(ModelBase):
    def __init__(self, num_inputs):
        self.input_shape = [num_inputs]

    def build_model(self, inputs):
        model = keras.models.Sequential([
            keras.layers.Dense(10, activation="selu", input_shape=inputs),
            keras.layers.Dense(5, activation="selu"),
            keras.layers.Dense(1, activation="relu")
        ])
        return model
