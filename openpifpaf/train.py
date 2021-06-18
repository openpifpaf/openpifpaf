"""Train a pifpaf network."""

import argparse
import copy
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

    # Slurm jobs might be stuck in the job queue and then started at exactly the
    # same time. Therefore we disambiguate with the Slurm job id.
    if os.getenv('SLURM_JOB_ID'):
        out += f'-slurm{os.getenv("SLURM_JOB_ID")}'

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
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.train',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--ddp', default=False, action='store_true',
                        help='[experimental] DistributedDataParallel')
    parser.add_argument('--local_rank', default=None, type=int,
                        help='[experimental] for torch.distributed.launch')
    parser.add_argument('--no-sync-batchnorm', dest='sync_batchnorm',
                        default=True, action='store_false',
                        help='[experimental] in ddp, to not use syncbatchnorm')

    logger.cli(parser)
    network.Factory.cli(parser)
    network.losses.Factory.cli(parser)
    network.Trainer.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    datasets.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    args = parser.parse_args()

    logger.configure(args, LOG)
    if args.log_stats:
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)

    # DDP with SLURM
    slurm_process_id = os.environ.get('SLURM_PROCID')
    if args.ddp and slurm_process_id is not None:
        if torch.cuda.device_count() > 1:
            LOG.warning('Expected one GPU per SLURM task but found %d. '
                        'Try with "srun --gpu-bind=closest ...". Still trying.',
                        torch.cuda.device_count())

        # if there is more than one GPU available, assume that other SLURM tasks
        # have access to the same GPUs and assign GPUs uniquely by slurm_process_id
        args.local_rank = (int(slurm_process_id) % torch.cuda.device_count()
                           if torch.cuda.device_count() > 0 else 0)

        os.environ['RANK'] = slurm_process_id
        if not os.environ.get('WORLD_SIZE') and os.environ.get('SLURM_NTASKS'):
            os.environ['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS')

        LOG.info('found SLURM process id: %s', slurm_process_id)
        LOG.info('distributed env: master=%s port=%s rank=%s world=%s, '
                 'local rank (GPU)=%d',
                 os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT'),
                 os.environ.get('RANK'), os.environ.get('WORLD_SIZE'),
                 args.local_rank)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    # output
    if args.output is None:
        args.output = default_output_file(args)
        os.makedirs('outputs', exist_ok=True)

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
    loss = network.losses.Factory().factory(datamodule.head_metas)

    checkpoint_shell = None
    if not args.disable_cuda and torch.cuda.device_count() > 1 and not args.ddp:
        LOG.info('Multiple GPUs with DataParallel: %d', torch.cuda.device_count())
        checkpoint_shell = copy.deepcopy(net_cpu)
        net = torch.nn.DataParallel(net_cpu.to(device=args.device))
        loss = loss.to(device=args.device)
    elif not args.disable_cuda and torch.cuda.device_count() == 1 and not args.ddp:
        LOG.info('Single GPU training')
        checkpoint_shell = copy.deepcopy(net_cpu)
        net = net_cpu.to(device=args.device)
        loss = loss.to(device=args.device)
    elif not args.disable_cuda and torch.cuda.device_count() > 0:
        LOG.info('Multiple GPUs with DistributedDataParallel')
        assert not list(loss.parameters())
        assert torch.cuda.device_count() > 0
        checkpoint_shell = copy.deepcopy(net_cpu)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        LOG.info('DDP: rank %d, world %d',
                 torch.distributed.get_rank(), torch.distributed.get_world_size())
        if args.sync_batchnorm:
            LOG.info('convert all batchnorms to syncbatchnorms')
            net_cpu = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_cpu)
        else:
            LOG.info('not converting batchnorms to syncbatchnorms')
        net = torch.nn.parallel.DistributedDataParallel(
            net_cpu.to(device=args.device),
            device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=isinstance(datamodule, datasets.MultiDataModule),
        )
        loss = loss.to(device=args.device)
    else:
        net = net_cpu

    logger.train_configure(args)
    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()
    if torch.distributed.is_initialized():
        train_loader = datamodule.distributed_sampler(train_loader)
        val_loader = datamodule.distributed_sampler(val_loader)

    optimizer = optimize.factory_optimizer(
        args, list(net.parameters()) + list(loss.parameters()))
    lr_scheduler = optimize.factory_lrscheduler(
        args, optimizer, len(train_loader), last_epoch=start_epoch)
    trainer = network.Trainer(
        net, loss, optimizer, args.output,
        checkpoint_shell=checkpoint_shell,
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
