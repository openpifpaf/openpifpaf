import logging
import os
from typing import Tuple

import torch
import torchvision

from .. import headmeta
from . import basenetworks, heads, nets

# generate hash values with: shasum -a 256 filename.pkl

PRETRAINED_UNAVAILABLE = object()

# Dataset cocokp is implied. All other datasets need to be explicit.
CHECKPOINT_URLS = {
    'resnet18': PRETRAINED_UNAVAILABLE,
    'resnet50': ('https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                 'v0.12a1/resnet50-200912-124649-cocokp-o10s-cda25db6.pkl'),
    'resnet101': PRETRAINED_UNAVAILABLE,
    'resnet152': PRETRAINED_UNAVAILABLE,
    'shufflenetv2x1': PRETRAINED_UNAVAILABLE,
    'shufflenetv2x2': PRETRAINED_UNAVAILABLE,
    'shufflenetv2k16': ('https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                        'v0.12a1/shufflenetv2k16w-200905-113300-cocokp-e7b371e6.pkl'),
    'shufflenetv2k16w': ('https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                         'v0.12a1/shufflenetv2k16w-200905-113300-cocokp-e7b371e6.pkl'),
    'shufflenetv2k30': ('https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                        'v0.12a1/shufflenetv2k30w-200918-223448-cocokp-o10s-be714792.pkl'),
    'shufflenetv2k44': PRETRAINED_UNAVAILABLE,
}

BASE_TYPES = set([basenetworks.Resnet, basenetworks.ShuffleNetV2, basenetworks.ShuffleNetV2K])
BASE_FACTORIES = {
    'resnet18': lambda: basenetworks.Resnet('resnet18', torchvision.models.resnet18, 512),
    'resnet50': lambda: basenetworks.Resnet('resnet50', torchvision.models.resnet50),
    'resnet101': lambda: basenetworks.Resnet('resnet101', torchvision.models.resnet101),
    'resnet152': lambda: basenetworks.Resnet('resnet152', torchvision.models.resnet152),
    'resnext50': lambda: basenetworks.Resnet('resnext50', torchvision.models.resnext50_32x4d),
    'resnext101': lambda: basenetworks.Resnet('resnext101', torchvision.models.resnext101_32x8d),
    'shufflenetv2x1': lambda: basenetworks.ShuffleNetV2(
        'shufflenetv2x1', torchvision.models.shufflenet_v2_x1_0, 1024),
    'shufflenetv2x2': lambda: basenetworks.ShuffleNetV2(
        'shufflenetv2x2', torchvision.models.shufflenet_v2_x2_0),
    'shufflenetv2k16': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k16', [4, 8, 4], [24, 348, 696, 1392, 1392]),
    'shufflenetv2k16w': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k16', [4, 8, 4], [24, 348, 696, 1392, 1392]),
    'shufflenetv2k20': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k20', [5, 10, 5], [32, 512, 1024, 2048, 2048]),
    'shufflenetv2k30': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k30', [8, 16, 6], [32, 512, 1024, 2048, 2048]),
    'shufflenetv2k44': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k44', [12, 24, 8], [32, 512, 1024, 2048, 2048]),
}

HEAD_TYPES = set([heads.CompositeField3])
HEAD_FACTORIES = {
    headmeta.Cif: heads.CompositeField3,
    headmeta.Caf: heads.CompositeField3,
    headmeta.CifDet: heads.CompositeField3,
}

LOG = logging.getLogger(__name__)


def factory_from_args(args, *, head_metas=None):
    return factory(
        checkpoint=args.checkpoint,
        base_name=args.basenet,
        head_metas=head_metas,
        dense_coupling=args.dense_coupling,
        cross_talk=args.cross_talk,
        two_scale=args.two_scale,
        multi_scale=args.multi_scale,
        multi_scale_hflip=args.multi_scale_hflip,
        download_progress=args.download_progress,
        head_consolidation=args.head_consolidation,
    )


def local_checkpoint_path(checkpoint):
    if os.path.exists(checkpoint):
        return checkpoint

    if checkpoint in CHECKPOINT_URLS:
        url = CHECKPOINT_URLS[checkpoint]

        base_dir = os.path.join(
            os.getenv('XDG_CACHE_HOME', os.path.join(os.getenv('HOME'), '.cache')),
            'torch',
        )
        if hasattr(torch, 'hub') and hasattr(torch.hub, 'get_dir'):
            # new in pytorch 1.6.0
            base_dir = torch.hub.get_dir()
        file_name = os.path.join(
            base_dir,
            'checkpoints',
            os.path.basename(url),
        )
        print(file_name, url, os.path.basename(url))

        if os.path.exists(file_name):
            return file_name

    return None


# pylint: disable=too-many-branches,too-many-statements
def factory(
        *,
        checkpoint=None,
        base_name=None,
        head_metas=None,
        dense_coupling=0.0,
        cross_talk=0.0,
        two_scale=False,
        multi_scale=False,
        multi_scale_hflip=True,
        download_progress=True,
        head_consolidation='filter_and_extend',
) -> Tuple[nets.Shell, int]:

    if base_name:
        assert head_metas
        assert checkpoint is None
        net_cpu: nets.Shell = factory_from_scratch(base_name, head_metas)
        epoch = 0
    else:
        assert base_name is None

        if not checkpoint:
            checkpoint = 'shufflenetv2k16w'

        if CHECKPOINT_URLS.get(checkpoint, None) is PRETRAINED_UNAVAILABLE:
            raise Exception(
                'The pretrained model for {} is not available yet '
                'in this release cycle. Use one of {}.'.format(
                    checkpoint,
                    [k for k, v in CHECKPOINT_URLS.items() if v is not PRETRAINED_UNAVAILABLE],
                )
            )
        checkpoint = CHECKPOINT_URLS.get(checkpoint, checkpoint)

        if checkpoint.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint,
                check_hash=not checkpoint.startswith('https'),
                progress=download_progress)
        else:
            checkpoint = torch.load(checkpoint)

        net_cpu: nets.Shell = checkpoint['model']
        epoch = checkpoint['epoch']

        # normalize for backwards compatibility
        nets.model_migration(net_cpu)

        if head_metas is not None and head_consolidation == 'keep':
            LOG.info('keeping heads from loaded checkpoint')
            # Match head metas by name and overwrite with meta from checkpoint.
            # This makes sure that the head metas have their head_index and
            # base_stride attributes set.
            input_head_meta_indices = {(meta.dataset, meta.name): i
                                       for i, meta in enumerate(head_metas)}
            for hn in net_cpu.head_nets:
                input_index = input_head_meta_indices.get((hn.meta.dataset, hn.meta.name), None)
                if input_index is None:
                    continue
                head_metas[input_index] = hn.meta
        elif head_metas is not None and head_consolidation == 'create':
            LOG.info('creating new heads')
            headnets = [HEAD_FACTORIES[h.__class__](h, net_cpu.base_net.out_features)
                        for h in head_metas]
            net_cpu.set_head_nets(headnets)
        elif head_metas is not None and head_consolidation == 'filter_and_extend':
            LOG.info('filtering for dataset heads and extending existing heads')
            existing_headnets = {(hn.meta.dataset, hn.meta.name): hn
                                 for hn in net_cpu.head_nets}
            headnets = []
            for meta_i, meta in enumerate(head_metas):
                if (meta.dataset, meta.name) in existing_headnets:
                    hn = existing_headnets[(meta.dataset, meta.name)]
                    headnets.append(hn)
                    # Match head metas by name and overwrite with meta from checkpoint.
                    # This makes sure that the head metas have their head_index and
                    # base_stride attributes set.
                    head_metas[meta_i] = hn.meta
                else:
                    headnets.append(
                        HEAD_FACTORIES[meta.__class__](meta, net_cpu.base_net.out_features))
            net_cpu.set_head_nets(headnets)
        elif head_metas is not None:
            raise Exception('head strategy {} unknown'.format(head_consolidation))

        # initialize for eval
        net_cpu.eval()

    if dense_coupling and not multi_scale:
        dcaf_meta = net_cpu.head_nets[2].meta
        dcaf_meta.decoder_confidence_scales = [dense_coupling for _ in dcaf_meta.skeleton]
        concatenated_caf = heads.CafConcatenate(
            (net_cpu.head_nets[1], net_cpu.head_nets[2]))
        net_cpu.set_head_nets([net_cpu.head_nets[0], concatenated_caf])
    elif dense_coupling and multi_scale:
        # TODO: fix multi-scale
        # cif_indices = [v * 3 + 1 for v in range(10)]
        # caf_indices = [v * 3 + 2 for v in range(10)]
        raise NotImplementedError
    if cross_talk:
        net_cpu.process_input = nets.CrossTalk(cross_talk)

    if two_scale:
        net_cpu = nets.Shell2Scale(net_cpu.base_net, net_cpu.head_nets)

    if multi_scale:
        net_cpu = nets.ShellMultiScale(net_cpu.base_net, net_cpu.head_nets,
                                       process_heads=net_cpu.process_heads,
                                       include_hflip=multi_scale_hflip)

    return net_cpu, epoch


def factory_from_scratch(basename, head_metas) -> nets.Shell:
    if basename not in BASE_FACTORIES:
        raise Exception('basename {} unknown'.format(basename))

    basenet = BASE_FACTORIES[basename]()
    headnets = [HEAD_FACTORIES[h.__class__](h, basenet.out_features) for h in head_metas]

    net_cpu = nets.Shell(basenet, headnets)
    nets.model_defaults(net_cpu)
    LOG.debug(net_cpu)
    return net_cpu


def configure(args):
    for bn in BASE_TYPES:
        bn.configure(args)
    for hn in HEAD_TYPES:
        hn.configure(args)


def cli(parser):
    for bn in BASE_TYPES:
        bn.cli(parser)
    for hn in HEAD_TYPES:
        hn.cli(parser)

    group = parser.add_argument_group('network configuration')
    group.add_argument('--checkpoint', default=None,
                       help=('Load a model from a checkpoint. '
                             'Use "resnet50", "shufflenetv2k16w" '
                             'or "shufflenetv2k30w" for pretrained OpenPifPaf models.'))
    group.add_argument('--basenet', default=None,
                       help='base network, e.g. resnet50')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--multi-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--no-multi-scale-hflip',
                       dest='multi_scale_hflip', default=True, action='store_false',
                       help='[experimental]')
    group.add_argument('--cross-talk', default=0.0, type=float,
                       help='[experimental]')
    group.add_argument('--no-download-progress', dest='download_progress',
                       default=True, action='store_false',
                       help='suppress model download progress bar')
    group.add_argument('--dense-coupling', default=0.0, type=float,
                       help='dense coupling')
    group.add_argument('--head-consolidation',
                       choices=('keep', 'create', 'filter_and_extend'),
                       default='filter_and_extend',
                       help=('consolidation strategy for a checkpoint\'s head '
                             'networks and the heads specified by the datamodule'))
