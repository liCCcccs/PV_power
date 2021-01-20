import logging
from importlib import import_module


class PluginLoader():
    @staticmethod
    def get_inspector(name, disable_logging=False):
        return PluginLoader._import("inspect.location", name, disable_logging)

    @staticmethod
    def get_model(name, disable_logging=False):
        # name should be eg.naive_NN
        return PluginLoader._import("train.model", name, disable_logging)

    @staticmethod
    def get_trainer(name, disable_logging=False):
        return PluginLoader._import("train.trainer", name, disable_logging)

    @staticmethod
    def _import(attr, name, disable_logging):
        ttl = attr.split(".")[-1].title()  # ttl = Model
        mod = ".".join(["plugins", attr.lower(), name])
        module = import_module(mod)
        return getattr(module, ttl)
