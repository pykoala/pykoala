import argparse
import sys

from logbook.compat import redirected_warnings, redirected_logging

from . import __version__ as koala_version
from .logging import logging_options, log_handler

DESCRIPTION = "<Add description here>"
REQUEST_CITATION_TEXT = "Please cite <TODO> if you use koala."
CITATION_TEXT = "<Add bibtex here>"


class CitationAction(argparse.Action):
    """
    A citation action, which prints the citation information.
    """

    def __init__(
        self, option_strings, citation=None, dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help="show program's bibtex citation and exit", **kwargs
    ):
        super(CitationAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0,
            help=help, **kwargs
        )
        if citation is None:
            raise ValueError("A bibtex citation must be provided.")
        self.citation = citation

    def __call__(self, parser, namespace, values, option_string=None):
        parser.exit(self.citation)


def parse_reduce_koala_data_cli(argv):
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=REQUEST_CITATION_TEXT,
    )
    logging_options(parser)
    parser.add_argument(
        '--version', action='version', version='%(prog)s ' + koala_version
    )
    parser.add_argument(
        '--citation', action=CitationAction, citation=CITATION_TEXT,
    )
    raise NotImplementedError("cli parser not written")
    return parser.parse_args(argv)


def parse_reduce_koala_data_config(args):
    raise NotImplementedError(
        "This should read in the yaml file and return a config object"
    )


def reduce_koala_data(config):
    raise NotImplementedError(
        "This should call the reduction steps in the correct order, based on "
        "the values in config"
    )


def reduce_koala_data_main(argv=None):
    """
    Main entry point for reduce-koala-data
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_reduce_koala_data_cli(argv)

    with log_handler(args), redirected_warnings(), redirected_logging():
        config = parse_reduce_koala_data_config(args)

        reduce_koala_data(config)
