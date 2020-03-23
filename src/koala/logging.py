# -*- coding: utf-8 -*-
"""
Logging configuration functions
"""
from logbook import NullHandler, FileHandler, NestedSetup
from logbook.more import ColorizedStderrHandler
from logbook.queues import ThreadedWrapperHandler


def logging_options(parser):
    """Add cli options for logging to parser"""
    LOG_LEVELS = ("critical", "error", "warning", "notice", "info", "debug")
    parser.add_argument("--log-file")
    parser.add_argument(
        "--log-file-level", choices=LOG_LEVELS, default="debug"
    )
    stderr_parser = parser.add_mutually_exclusive_group()
    stderr_parser.add_argument(
        "--stderr-level", choices=LOG_LEVELS, default="notice"
    )
    stderr_parser.add_argument(
        "--quiet", "-q", default=False, action="store_true",
    )
    stderr_parser.add_argument(
        "--verbose", "-v", default=False, action="store_true",
    )


def log_handler(args, thread_wrapping=True):
    """
    Return log handler with given config
    """
    if not isinstance(args, dict):
        args = vars(args)
    if args.get("quiet"):
        stderr_handler = ColorizedStderrHandler(level="ERROR")
    elif args.get("verbose"):
        stderr_handler = ColorizedStderrHandler(level="DEBUG")
    else:
        stderr_handler = ColorizedStderrHandler(
            level=args.get("stderr_level", "NOTICE").upper(), bubble=True
        )
    if args.get("log_file"):
        file_handler = FileHandler(
            args.get("log_file"),
            level=args.get("log_file_level", "DEBUG").upper(), bubble=True
        )
    else:
        file_handler = NullHandler()

    if thread_wrapping:
        file_handler = ThreadedWrapperHandler(file_handler)
        stderr_handler = ThreadedWrapperHandler(stderr_handler)

    return NestedSetup([
        NullHandler(),  # catch everything else
        file_handler, stderr_handler
    ])
