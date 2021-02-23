import argparse
import logging
import socket
import sys

import torch

LOG = logging.getLogger(__name__)


def cli(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('logger')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')


def configure(args: argparse.Namespace, local_logger=None):
    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        assert not args.quiet
        log_level = logging.DEBUG

    if args.log_stats:
        # pylint: disable=import-outside-toplevel
        from pythonjsonlogger import jsonlogger
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            jsonlogger.JsonFormatter('(message) (levelname) (name)'))
        logging.basicConfig(handlers=[stdout_handler])
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)
    else:
        logging.basicConfig()

    # set log level for openpifpaf and all openpifpaf plugins
    for logger_name in logging.root.manager.loggerDict:  # pylint: disable=no-member
        if '.' in logger_name or not logger_name.startswith('openpifpaf'):
            continue
        logging.getLogger(logger_name).setLevel(log_level)

    if local_logger is not None:
        local_logger.setLevel(log_level)


def train_configure(args):
    if torch.distributed.is_initialized():
        rank_prefix = 'Rank {}/{}'.format(
            torch.distributed.get_rank(), torch.distributed.get_world_size())
        formatter = logging.Formatter(rank_prefix + ' - %(levelname)s:%(name)s:%(message)s')
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    # pylint: disable=import-outside-toplevel,cyclic-import
    from pythonjsonlogger import jsonlogger
    from . import __version__
    from .plugin import versions as plugin_versions

    file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(
        jsonlogger.JsonFormatter('%(message) %(levelname) %(name) %(asctime)'))
    logging.getLogger('openpifpaf').addHandler(file_handler)

    LOG.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': __version__,
        'plugin_versions': plugin_versions(),
        'hostname': socket.gethostname(),
    })
