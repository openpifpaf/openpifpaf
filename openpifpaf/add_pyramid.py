import argparse
import datetime
import logging
import socket

import torch

from . import __version__ as VERSION
from .network import heads
from .network import nets

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level)
    LOG.setLevel(level)
    return args


def add_pyramid(net_cpu):
    if net_cpu.base_net.shortname.startswith('shufflenet'):
        LOG.debug('processing a shufflenet model')
        nets.shufflenet_add_pyramid(net_cpu.base_net)
    elif net_cpu.base_net.shortname.startswith('resnet'):
        LOG.debug('processing a resnet model')
        nets.resnet_add_pyramid(net_cpu.base_net)
    else:
        raise Exception('cannot transform base net {}'.format(net_cpu.base_net.shortname))
    net_cpu.base_net.shortname += 'pd'

    LOG.debug('recreating head nets')
    head_nets = [heads.factory(h, net_cpu.base_net.out_features)
                 for h in net_cpu.head_names]

    net_cpu = nets.Shell(net_cpu.base_net, head_nets)
    nets.model_defaults(net_cpu)
    return net_cpu


def main():
    args = cli()
    net_cpu, epoch = nets.factory_from_args(args)

    net_cpu = add_pyramid(net_cpu)

    # write model
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    basename = net_cpu.base_net.shortname
    headnames = '-'.join(net_cpu.head_names)
    filename = 'outputs/{}-{}-{}.pkl'.format(basename, headnames, now)
    LOG.debug('about to write model %s', filename)
    torch.save({
        'model': net_cpu,
        'epoch': epoch,
        'meta': {
            'args': vars(args),
            'version': VERSION,
            'hostname': socket.gethostname(),
        },
    }, filename)
    LOG.info('model written: %s', filename)



if __name__ == '__main__':
    main()
