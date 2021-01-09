import os
from importlib import import_module

class ScriptExecutor():
    def __init__(self, command):
        self._command = command.lower()

    def _import_script(self):
        mod = ".".join(self._command, "NN_2in_1out")   # only test this first, command = naive_NN
                                                        #TODO: expand this
        module = import_module(mod)
        script = getattr(module, self._command.title())   # should be importing the class from the script
        return script

    def execute_script(self, arguments):
        pass