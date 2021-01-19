import argparse
from .launcher import ScriptExecutor


class PVArgs():
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
        return global_args

    def _add_arguments(self):
        """ Parse the list of dictionaries containing the command line arguments and convert to
        argparse parser arguments. """
        options = self._get_global_arguments + self.argument_list
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)


class TrainArgs(PVArgs):
    @staticmethod
    def get_argument_list():
        argument_list = list()
        argument_list.append(dict(
            opts=("-dir", "--data_dir"),
            dest="data_dir",
            default="./data/naive_NN/",
            group="Global Options",
            help="Specify the model you want to use"))
        argument_list.append(dict(
            opts=("-model", "--model"),
            dest="model",
            default="naive_NN",
            group="Global Options",
            help="Specify the model you want to use"))
        argument_list.append(dict(
            opts=("-ep", "--epochs"),
            dest="epochs",
            default=10,
            group="Global Options",
            help="Specify the model you want to use"))
        argument_list.append(dict(
            opts=("-bs", "--batch_size"),
            dest="batch_size",
            default=16,
            group="Global Options",
            help="Batch size"))
        return argument_list


class InspectArgs(PVArgs):
    @staticmethod
    def get_argument_list():
        argument_list = list()
        argument_list.append(dict(
            opts=("-fp", "--file_path"),
            dest="file_path",
            default="./data/Yulara",
            group="Plugins",
            help="Specify the file path of the data"))
        argument_list.append(dict(
            opts=("-o", "--option"),
            dest="inspect_option",
            default=2,
            type=int,
            group="Plugins",
            help="1: plot power and ..."))
        argument_list.append(dict(
            opts=("-s", "--start_date"),
            dest="start_date",
            default="2016/4/7",
            group="Plugins",
            help="Specify the start date"))
        argument_list.append(dict(
            opts=("-e", "--end_date"),
            dest="end_date",
            default="2016/4/11",
            group="Plugins",
            help="Specify the end date"))
        return argument_list
