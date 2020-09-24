"""Train a pifpaf network."""

import argparse
import datetime
import logging
import os
import socket

import torch

from . import datasets, encoder, logger, network, optimize, plugins, show, visualizer
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
    plugins.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.train',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    logger.cli(parser)
    network.cli(parser)
    network.losses.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    datasets.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--val-interval', default=1, type=int,
                        help='validation run every n epochs')
    parser.add_argument('--train-batches', default=None, type=int,
                        help='number of train batches')
    parser.add_argument('--val-batches', default=None, type=int,
                        help='number of val batches')
    parser.add_argument('--fix-batch-norm',
                        default=False, action='store_true',
                        help='fix batch norm running statistics')
    parser.add_argument('--ema', default=1e-2, type=float,
                        help='ema decay constant')
    parser.add_argument('--clip-grad-norm', default=0.0, type=float,
                        help='clip grad norm: specify largest change for single param')
    parser.add_argument('--log-interval', default=11, type=int,
                        help='log loss every n steps')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')

    group = parser.add_argument_group('debug')
    group.add_argument('--profile', default=None,
                       help='enables profiling. specify path for chrome tracing file')

    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

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

    logger.train_configure(args, LOG)
    if args.log_stats:
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)

    network.configure(args)
    network.losses.configure(args)
    encoder.configure(args)
    datasets.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def main():
    args = cli()

    datamodule = datasets.factory(args.dataset)

    net_cpu, start_epoch = network.factory_from_args(args, head_metas=datamodule.head_metas)
    net = net_cpu.to(device=args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        print('Using multiple GPUs: {}'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    loss = network.losses.factory_from_args(args, net_cpu.head_nets)
    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()

    optimizer = optimize.factory_optimizer(
        args, list(net.parameters()) + list(loss.parameters()))
    lr_scheduler = optimize.factory_lrscheduler(args, optimizer, len(train_loader))
    trainer = network.Trainer(
        net, loss, optimizer, args.output,
        lr_scheduler=lr_scheduler,
        device=args.device,
        fix_batch_norm=args.fix_batch_norm,
        stride_apply=args.stride_apply,
        ema_decay=args.ema,
        log_interval=args.log_interval,
        train_profile=args.profile,
        model_meta_data={
            'args': vars(args),
            'version': __version__,
            'hostname': socket.gethostname(),
        },
        clip_grad_norm=args.clip_grad_norm,
        val_interval=args.val_interval,
        n_train_batches=args.train_batches,
        n_val_batches=args.val_batches,
    )
    trainer.loop(train_loader, val_loader, args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
