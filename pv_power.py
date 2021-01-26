import argparse
import sys
from lib import args


_PARSER = argparse.ArgumentParser()


def _bad_args(*args):  # pylint:disable=unused-argument
    """ Print help to console when bad arguments are provided. """
    print(args)
    _PARSER.print_help()
    sys.exit(0)


def _main():
    """ The main entry point of pv_power """
    subparser = _PARSER.add_subparsers()
    args.TrainArgs(subparser, "train")
    args.InspectArgs(subparser, "inspect")

    _PARSER.set_defaults()
    arguments = _PARSER.parse_args(["inspect",
                                    "-fp", "./data/Yulara",
                                    "-o", "2",
                                    "-s", "2020/09/01",
                                    "-e", "2020/11/29"])
    arguments.func(arguments)


if __name__ == "__main__":
    _main()