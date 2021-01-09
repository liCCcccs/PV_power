import argparse
from .launcher import ScriptExecutor

class TrainArgs():
    def __init__(self, subparser, command, description="default"):
        self.argument_list = self.get_argument_list()
        if not subparser:
            return
        self.parser = self._create_parser(subparser, command, description)
        self._add_arguments()
        script = ScriptExecutor(command)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_argument_list():
        argument_list = []
        return argument_list

    @staticmethod
    def _create_parser(subparser, command, description):
        parser = subparser.add_parser(command,
                                      help=description)
        return parser

    def _add_arguments(self):
        """ Parse the list of dictionaries containing the command line arguments and convert to
        argparse parser arguments. """
        options = self.argument_list
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)
