import argparse
from .launcher import ScriptExecutor


class TrainArgs():
    def __init__(self, subparser, command, description="default"):
        self._get_global_arguments = self._get_global_arguments()
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

    @staticmethod
    def _get_global_arguments():
        global_args = list()
        global_args.append(dict(
            opts=("-arg1", "--arg1"),
            dest="test_arg1",
            default="NN2in2out",
            group="Global Options",
            help="Specify the model you want to use"))
        global_args.append(dict(
            opts=("-arg2", "--arg2"),
            dest="test_arg2",
            default="NN2in2out",
            group="Global Options",
            help="Specify the model you want to use"))
        global_args.append(dict(
            opts=("-bs", "--batch_size"),
            dest="batch_size",
            default=16,
            group="Global Options",
            help="Batch size"))
        return global_args

    def _add_arguments(self):
        """ Parse the list of dictionaries containing the command line arguments and convert to
        argparse parser arguments. """
        options = self._get_global_arguments + self.argument_list
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)

