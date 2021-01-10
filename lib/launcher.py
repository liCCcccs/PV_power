import os
from importlib import import_module
import logging


class ScriptExecutor():
    def __init__(self, command):
        self._command = command.lower()

    def _import_script(self):
        mod = ".".join(["script", self._command])   # only test this first, command = naive_NN
                                                        #TODO: expand this
        module = import_module(mod)
        script = getattr(module, self._command.title())   # should be importing the class from the script
        return script

    def execute_script(self, arguments):
        script = self._import_script()
        process = script(arguments)  # initialize that class with arguments
        process.process()  # execute the process() function in the class
        """
        try:
            script = self._import_script()
            process = script(arguments)   # initialize that class with arguments
            process.process()   # execute the process() function in the class
        except KeyboardInterrupt:
            raise
        except Exception:
            logging.warning("the main function has an error")
        """