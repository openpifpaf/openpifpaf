import logging
import os
import torch

import torchvision

from .. import headmeta
from . import basenetworks, heads, nets

# generate hash values with: shasum -a 256 filename.pkl


CHECKPOINT_URLS = {
    'resnet50': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                 'v0.11.2/resnet50-200527-171310-cif-caf-caf25-o10s-c0b7ae80.pkl'),
    'shufflenetv2k16w': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                         'v0.11.0/shufflenetv2k16w-200510-221334-cif-caf-caf25-o10s-604c5956.pkl'),
    'shufflenetv2k30w': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                         'v0.11.0/shufflenetv2k30w-200510-104256-cif-caf-caf25-o10s-0b5ba06f.pkl'),
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
    'shufflenetv2k16w': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k16w', [4, 8, 4], [24, 348, 696, 1392, 1392]),
    'shufflenetv2k20w': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k20w', [5, 10, 5], [32, 512, 1024, 2048, 2048]),
    'shufflenetv2k30w': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k30w', [8, 16, 6], [32, 512, 1024, 2048, 2048]),
    'shufflenetv2k44w': lambda: basenetworks.ShuffleNetV2K(
        'shufflenetv2k44w', [12, 24, 8], [32, 512, 1024, 2048, 2048]),
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
        download_progress=True):

    if base_name:
        assert head_metas
        assert checkpoint is None
        net_cpu = factory_from_scratch(base_name, head_metas)
        epoch = 0
    else:
        assert base_name is None
        assert head_metas is None

        if not checkpoint:
            checkpoint = 'shufflenetv2k16w'

        if checkpoint == 'resnet18':
            raise Exception('this pretrained model is currently not available')
        if checkpoint == 'resnet101':
            raise Exception('this pretrained model is currently not available')
        checkpoint = CHECKPOINT_URLS.get(checkpoint, checkpoint)

        if checkpoint.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint,
                check_hash=not checkpoint.startswith('https'),
                progress=download_progress)
        else:
            checkpoint = torch.load(checkpoint)

        net_cpu = checkpoint['model']
        epoch = checkpoint['epoch']

        # normalize for backwards compatibility
        nets.model_migration(net_cpu)

        # initialize for eval
        net_cpu.eval()

    if dense_coupling and not multi_scale:
        dcaf_meta = net_cpu.head_nets[2].meta
        dcaf_meta.decoder_confidence_scales = [dense_coupling for _ in dcaf_meta.skeleton]
        concatenated_caf = heads.CafConcatenate(
            (net_cpu.head_nets[1], net_cpu.head_nets[2]))
        net_cpu.head_nets = torch.nn.ModuleList([net_cpu.head_nets[0], concatenated_caf])
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


def factory_from_scratch(basename, head_metas):
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
