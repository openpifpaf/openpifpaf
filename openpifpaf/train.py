"""Train a pifpaf network."""

import argparse
import datetime
import logging
import socket

import torch

from . import datasets, encoder, logs, network, optimize, visualizer
from . import __version__

LOG = logging.getLogger(__name__)


def default_output_file(args, net_cpu):
    base_name = net_cpu.base_net.shortname
    head_names = [hn.meta.name for hn in net_cpu.head_nets]

    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out = 'outputs/{}-{}-{}'.format(base_name, now, '-'.join(head_names))
    if args.square_edge != 385:
        out += '-edge{}'.format(args.square_edge)
    if args.regression_loss != 'laplace':
        out += '-{}'.format(args.regression_loss)
    if args.r_smooth != 0.0:
        out += '-rsmooth{}'.format(args.r_smooth)
    if args.orientation_invariant or args.extended_scale:
        out += '-'
        if args.orientation_invariant:
            out += 'o{:02.0f}'.format(args.orientation_invariant * 100.0)
        if args.extended_scale:
            out += 's'

    return out + '.pkl'


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.train',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    logs.cli(parser)
    network.cli(parser)
    network.losses.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    datasets.train_cli(parser)
    visualizer.cli(parser)

    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--rescale-images', type=float, default=1.0,
                        help='overall image rescale factor')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')

    group = parser.add_argument_group('debug')
    group.add_argument('--profile', default=None,
                       help='enables profiling. specify path for chrome tracing file')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')

    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    network.configure(args)
    network.losses.configure(args)
    encoder.configure(args)
    datasets.train_configure(args)
    visualizer.configure(args)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    return args


def main():
    args = cli()
    net_cpu, start_epoch = network.factory_from_args(args)
    net_cpu.process_heads = None
    if args.output is None:
        args.output = default_output_file(args, net_cpu)
    logs.configure(args)
    if args.log_stats:
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)

    net = net_cpu.to(device=args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        print('Using multiple GPUs: {}'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    loss = network.losses.factory_from_args(args, net_cpu.head_nets)
    target_transforms = encoder.factory(net_cpu.head_nets, net_cpu.base_net.stride)
    train_loader, val_loader = datasets.train_factory(args, target_transforms)

    optimizer = optimize.factory_optimizer(
        args, list(net.parameters()) + list(loss.parameters()))
    lr_scheduler = optimize.factory_lrscheduler(args, optimizer, len(train_loader))
    trainer = network.Trainer(
        net, loss, optimizer, args.output,
        lr_scheduler=lr_scheduler,
        device=args.device,
        fix_batch_norm=not args.update_batchnorm_runningstatistics,
        stride_apply=args.stride_apply,
        ema_decay=args.ema,
        train_profile=args.profile,
        model_meta_data={
            'args': vars(args),
            'version': __version__,
            'hostname': socket.gethostname(),
        },
    )
    trainer.loop(train_loader, val_loader, args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
