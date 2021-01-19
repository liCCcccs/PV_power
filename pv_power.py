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
                                    "-o", "1",
                                    "-s", "2018/12/28",
                                    "-e", "2019/1/3"])
    arguments.func(arguments)
    # lili


if __name__ == "__main__":
    _main()