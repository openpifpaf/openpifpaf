"""Train a pifpaf network."""

import argparse
import datetime
import logging
import socket

import torch

from . import datasets, encoder, logs, optimize, transforms
from .network import losses, nets, Trainer
from . import __version__ as VERSION


def default_output_file(args, net_cpu):
    base_name = net_cpu.base_net.shortname
    head_names = net_cpu.head_names

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
            out += 'o'
        if args.extended_scale:
            out += 's'

    return out + '.pkl'


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    logs.cli(parser)
    nets.cli(parser)
    losses.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    datasets.train_cli(parser)

    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--rescale-images', type=float, default=1.0,
                        help='overall image rescale factor')
    parser.add_argument('--orientation-invariant', default=False, action='store_true',
                        help='augment with random orientations')
    parser.add_argument('--extended-scale', default=False, action='store_true',
                        help='augment with an extended scale range')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=385, type=int,
                        help='square edge of input images')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--no-augmentation', dest='augmentation',
                        default=True, action='store_false',
                        help='do not apply data augmentation')

    group = parser.add_argument_group('debug')
    group.add_argument('--profile', default=None,
                       help='enables profiling. specify path for chrome tracing file')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')

    args = parser.parse_args()

    encoder.configure(args)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def main():
    args = cli()
    net_cpu, start_epoch = nets.factory_from_args(args)
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

    loss = losses.factory_from_args(args, net_cpu.head_names)
    target_transforms = encoder.factory(
        net_cpu.head_names, net_cpu.head_strides, skeleton=args.debug)

    if args.augmentation:
        preprocess_transformations = [
            transforms.NormalizeAnnotations(),
            transforms.AnnotationJitter(),
            transforms.RandomApply(transforms.HFlip(), 0.5),
        ]
        if args.extended_scale:
            preprocess_transformations.append(
                transforms.RescaleRelative(scale_range=(0.25 * args.rescale_images,
                                                        2.0 * args.rescale_images),
                                           power_law=True)
            )
        else:
            preprocess_transformations.append(
                transforms.RescaleRelative(scale_range=(0.4 * args.rescale_images,
                                                        2.0 * args.rescale_images),
                                           power_law=True)
            )
        preprocess_transformations += [
            # transforms.RandomApply(transforms.ScaleMix(args.square_edge / 2.0), 0.5),
            transforms.Crop(args.square_edge),
            transforms.CenterPad(args.square_edge),
        ]
        if args.orientation_invariant:
            preprocess_transformations += [
                transforms.RotateBy90(),
            ]
        preprocess_transformations += [
            transforms.TRAIN_TRANSFORM,
        ]
    else:
        preprocess_transformations = [
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.square_edge),
            transforms.CenterPad(args.square_edge),
            transforms.EVAL_TRANSFORM,
        ]
    preprocess = transforms.Compose(preprocess_transformations)
    train_loader, val_loader = datasets.train_factory(
        args, preprocess, target_transforms)

    optimizer = optimize.factory_optimizer(
        args, list(net.parameters()) + list(loss.parameters()))
    lr_scheduler = optimize.factory_lrscheduler(args, optimizer, len(train_loader))
    encoder_visualizer = None
    if args.debug:
        encoder_visualizer = encoder.Visualizer(net_cpu.head_names, net_cpu.head_strides)

    trainer = Trainer(
        net, loss, optimizer, args.output,
        lr_scheduler=lr_scheduler,
        device=args.device,
        fix_batch_norm=not args.update_batchnorm_runningstatistics,
        stride_apply=args.stride_apply,
        ema_decay=args.ema,
        encoder_visualizer=encoder_visualizer,
        train_profile=args.profile,
        model_meta_data={
            'args': vars(args),
            'version': VERSION,
            'hostname': socket.gethostname(),
        },
    )
    trainer.loop(train_loader, val_loader, args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
