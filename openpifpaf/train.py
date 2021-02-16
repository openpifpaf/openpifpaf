"""Train a pifpaf network."""

import argparse
import datetime
import logging
import os
import socket

import torch

from . import datasets, encoder, logger, network, optimize, plugin, show, visualizer
from . import __version__

LOG = logging.getLogger(__name__)


def default_output_file(args):
    base_name = args.basenet
    if not base_name:
        base_name, _, __ = os.path.basename(args.checkpoint).partition('-')

    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out = 'outputs/{}-{}-{}'.format(base_name, now, args.dataset)
    if args.cocokp_square_edge != 385:
        out += '-edge{}'.format(args.cocokp_square_edge)
    if args.regression_loss != 'laplace':
        out += '-{}'.format(args.regression_loss)
    if args.r_smooth != 0.0:
        out += '-rsmooth{}'.format(args.r_smooth)
    if args.cocokp_orientation_invariant or args.cocokp_extended_scale:
        out += '-'
        if args.cocokp_orientation_invariant:
            out += 'o{:02.0f}'.format(args.cocokp_orientation_invariant * 100.0)
        if args.cocokp_extended_scale:
            out += 's'

    return out + '.pkl'


def cli():
    plugin.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.train',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))
    parser.add_argument('--ddp', default=False, action='store_true')
    parser.add_argument('--local_rank', type=int)

    logger.cli(parser)
    network.Factory.cli(parser)
    network.losses.Factory.cli(parser)
    network.Trainer.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    datasets.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')

    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # output
    if args.output is None:
        args.output = default_output_file(args)
        os.makedirs('outputs', exist_ok=True)

    logger.configure(args, LOG)
    if args.log_stats:
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)

    network.Factory.configure(args)
    network.losses.Factory.configure(args)
    network.Trainer.configure(args)
    encoder.configure(args)
    datasets.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def main():
    args = cli()

    datamodule = datasets.factory(args.dataset)

    net_cpu, start_epoch = network.Factory().factory(head_metas=datamodule.head_metas)
    loss = network.losses.Factory().factory(net_cpu.head_nets)

    if not args.ddp:
        net = net_cpu.to(device=args.device)
        if not args.disable_cuda and torch.cuda.device_count() > 1:
            LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
            net = torch.nn.DataParallel(net)
        loss = loss.to(device=args.device)
    else:
        assert torch.cuda.device_count() > 0
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')
        LOG.info('DDP: rank %d, world %d', torch.distributed.get_rank(), torch.distributed.get_world_size())
        net = torch.nn.parallel.DistributedDataParallel(net_cpu.to(device=args.device),
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank)

    logger.train_configure(args)
    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()

    optimizer = optimize.factory_optimizer(
        args, list(net.parameters()) + list(loss.parameters()))
    lr_scheduler = optimize.factory_lrscheduler(args, optimizer, len(train_loader))
    trainer = network.Trainer(
        net, loss, optimizer, args.output,
        lr_scheduler=lr_scheduler,
        device=args.device,
        model_meta_data={
            'args': vars(args),
            'version': __version__,
            'plugin_versions': plugin.versions(),
            'hostname': socket.gethostname(),
        },
    )
    trainer.loop(train_loader, val_loader, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
