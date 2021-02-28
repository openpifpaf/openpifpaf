import argparse
import logging
import os
from typing import Tuple
import warnings

import torch
import torchvision

from .. import headmeta
from ..configurable import Configurable
from . import basenetworks, heads, nets

# generate hash values with: shasum -a 256 filename.pkl

PRETRAINED_UNAVAILABLE = object()

# Dataset cocokp is implied. All other datasets need to be explicit.
# Use http instead of https to avoid SSL certificate issues on Windows.
CHECKPOINT_URLS = {
    'mobilenetv2': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                    'v0.12a5/mobilenetv2-201112-193315-cocokp-1728a9f5.pkl'),
    'resnet18': PRETRAINED_UNAVAILABLE,
    'resnet50': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                 'v0.12.2/resnet50-210224-202010-cocokp-o10s-d020d7f1.pkl'),
    'resnet101': PRETRAINED_UNAVAILABLE,
    'resnet152': PRETRAINED_UNAVAILABLE,
    'shufflenetv2x1': PRETRAINED_UNAVAILABLE,
    'shufflenetv2x2': PRETRAINED_UNAVAILABLE,
    'shufflenetv2k16': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                        'v0.12b4/shufflenetv2k16-210214-123448-cocokp-o10s-e2ae3708.pkl'),
    'shufflenetv2k16-withdense': (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12b4/shufflenetv2k16-210221-131426-cocokp-o10s-627d901e.pkl'
    ),
    'shufflenetv2k30': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                        'v0.12.2/shufflenetv2k30-210224-074128-cocokp-o10s-59ca2b89.pkl'),
    'shufflenetv2k44': PRETRAINED_UNAVAILABLE,
}

BASE_TYPES = set([
    basenetworks.MobileNetV2,
    basenetworks.Resnet,
    basenetworks.ShuffleNetV2,
    basenetworks.ShuffleNetV2K,
    basenetworks.SqueezeNet,
])
BASE_FACTORIES = {
    'mobilenetv2': lambda: basenetworks.MobileNetV2('mobilenetv2', torchvision.models.mobilenet_v2),
    'resnet18': lambda: basenetworks.Resnet('resnet18', torchvision.models.resnet18, 512),
    'resnet50': lambda: basenetworks.Resnet('resnet50', torchvision.models.resnet50),
    'resnet101': lambda: basenetworks.Resnet('resnet101', torchvision.models.resnet101),
    'resnet152': lambda: basenetworks.Resnet('resnet152', torchvision.models.resnet152),
    'resnext50': lambda: basenetworks.Resnet('resnext50', torchvision.models.resnext50_32x4d),
    'resnext101': lambda: basenetworks.Resnet('resnext101', torchvision.models.resnext101_32x8d),
    'shufflenetv2x1': lambda: basenetworks.ShuffleNetV2(
        'shufflenetv2x1', torchvision.models.shufflenet_v2_x1_0, 1024),
    'shufflenetv2x2': lambda: basenetworks.ShuffleNetV2(
        # defined in torchvision as [4, 8, 4], [24, 244, 488, 976, 2048]
        'shufflenetv2x2', torchvision.models.shufflenet_v2_x2_0),
    'shufflenetv2k16': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k16', [4, 8, 4], [24, 348, 696, 1392, 1392]),
    'shufflenetv2k20': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k20', [5, 10, 5], [32, 512, 1024, 2048, 2048]),
    'shufflenetv2kx5': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2kx5', [6, 13, 6], [42, 640, 1280, 2560, 2560]),
    'shufflenetv2k30': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k30', [8, 16, 6], [32, 512, 1024, 2048, 2048]),
    'shufflenetv2k44': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k44', [12, 24, 8], [32, 512, 1024, 2048, 2048]),
    'squeezenet': lambda: basenetworks.SqueezeNet('squeezenet', torchvision.models.squeezenet1_1),
}

#: headmeta class to head class
HEADS = {
    headmeta.Cif: heads.CompositeField3,
    headmeta.Caf: heads.CompositeField3,
    headmeta.CifDet: heads.CompositeField3,
}

LOG = logging.getLogger(__name__)


def local_checkpoint_path(checkpoint):
    if os.path.exists(checkpoint):
        return checkpoint

    if checkpoint in CHECKPOINT_URLS:
        url = CHECKPOINT_URLS[checkpoint]

        base_dir = None
        if hasattr(torch, 'hub') and hasattr(torch.hub, 'get_dir'):
            # new in pytorch 1.6.0
            base_dir = torch.hub.get_dir()
        elif os.getenv('TORCH_HOME'):
            base_dir = os.getenv('TORCH_HOME')
        elif os.getenv('XDG_CACHE_HOME'):
            base_dir = os.path.join(os.getenv('XDG_CACHE_HOME'), 'torch')
        else:
            base_dir = os.path.expanduser(os.path.join('~', '.cache', 'torch'))

        file_name = os.path.join(base_dir, 'checkpoints', os.path.basename(url))
        if os.path.exists(file_name):
            return file_name

    return None


class Factory(Configurable):
    base_name = None
    checkpoint = None
    cross_talk = 0.0
    download_progress = True
    head_consolidation = 'filter_and_extend'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.base_name is not None:
            assert self.checkpoint is None
        if self.checkpoint is not None:
            assert self.base_name is None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        for bn in BASE_TYPES:
            bn.cli(parser)
        for hn in set(HEADS.values()):
            hn.cli(parser)

        group = parser.add_argument_group('network configuration')
        available_checkpoints = ['"{}"'.format(n) for n, url in CHECKPOINT_URLS.items()
                                 if url is not PRETRAINED_UNAVAILABLE]
        group.add_argument(
            '--checkpoint', default=cls.checkpoint,
            help=(
                'Path to a local checkpoint. '
                'Or provide one of the following to download a pretrained model: {}'
                ''.format(', '.join(available_checkpoints))
            )
        )
        group.add_argument('--basenet', default=cls.base_name,
                           help='base network, e.g. resnet50')
        group.add_argument('--cross-talk', default=cls.cross_talk, type=float,
                           help='[experimental]')
        assert cls.download_progress
        group.add_argument('--no-download-progress', dest='download_progress',
                           default=True, action='store_false',
                           help='suppress model download progress bar')
        group.add_argument('--head-consolidation',
                           choices=('keep', 'create', 'filter_and_extend'),
                           default=cls.head_consolidation,
                           help=('consolidation strategy for a checkpoint\'s head '
                                 'networks and the heads specified by the datamodule'))

    @classmethod
    def configure(cls, args: argparse.Namespace):
        for bn in BASE_TYPES:
            bn.configure(args)
        for hn in set(HEADS.values()):
            hn.configure(args)

        cls.base_name = args.basenet
        cls.checkpoint = args.checkpoint
        cls.cross_talk = args.cross_talk
        cls.download_progress = args.download_progress
        cls.head_consolidation = args.head_consolidation

    def factory(self, *, head_metas=None) -> Tuple[nets.Shell, int]:
        if self.base_name:
            assert head_metas
            assert self.checkpoint is None
            net_cpu: nets.Shell = self.from_scratch(head_metas)
            net_cpu = self.init_net(net_cpu)
            epoch = 0
            return net_cpu, epoch

        net_cpu, epoch = self.from_checkpoint()
        if head_metas is not None:
            self.consolidate_heads(net_cpu, head_metas)

        net_cpu = self.init_net(net_cpu)
        return net_cpu, epoch

    def consolidate_heads(self, net_cpu, head_metas):
        if self.head_consolidation == 'keep':
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
        elif self.head_consolidation == 'create':
            LOG.info('creating new heads')
            headnets = [HEADS[h.__class__](h, net_cpu.base_net.out_features)
                        for h in head_metas]
            net_cpu.set_head_nets(headnets)
        elif self.head_consolidation == 'filter_and_extend':
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
                        HEADS[meta.__class__](meta, net_cpu.base_net.out_features))
            net_cpu.set_head_nets(headnets)
        else:
            raise Exception('head strategy {} unknown'.format(self.head_consolidation))

    def from_checkpoint(self) -> Tuple[nets.Shell, int]:
        checkpoint = self.checkpoint
        if not checkpoint:
            checkpoint = 'shufflenetv2k16'

        if CHECKPOINT_URLS.get(checkpoint, None) is PRETRAINED_UNAVAILABLE:
            raise Exception(
                'The pretrained model for {} is not available yet '
                'in this release cycle. Use one of {}.'.format(
                    checkpoint,
                    [k for k, v in CHECKPOINT_URLS.items() if v is not PRETRAINED_UNAVAILABLE],
                )
            )
        checkpoint = CHECKPOINT_URLS.get(checkpoint, checkpoint)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=torch.serialization.SourceChangeWarning)

            if checkpoint.startswith('http'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    checkpoint,
                    check_hash=not checkpoint.startswith('https'),
                    progress=self.download_progress)
            else:
                checkpoint = torch.load(checkpoint)

        net_cpu: nets.Shell = checkpoint['model']
        epoch = checkpoint['epoch']

        # normalize for backwards compatibility
        nets.model_migration(net_cpu)

        return net_cpu, epoch

    def from_scratch(self, head_metas) -> nets.Shell:
        if self.base_name not in BASE_FACTORIES:
            raise Exception('basenet {} unknown'.format(self.base_name))

        basenet = BASE_FACTORIES[self.base_name]()
        headnets = [HEADS[h.__class__](h, basenet.out_features) for h in head_metas]

        net_cpu = nets.Shell(basenet, headnets)
        nets.model_defaults(net_cpu)
        return net_cpu

    def init_net(self, net_cpu):
        if self.cross_talk:
            net_cpu.process_input = nets.CrossTalk(self.cross_talk)

        # initialize for eval
        net_cpu.eval()

        LOG.debug(net_cpu)
        return net_cpu
