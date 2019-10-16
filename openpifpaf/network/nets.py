import logging
import re
import torch
import torchvision

from . import basenetworks, heads
from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_CONNECTIONS, HFLIP
from .. import torch_hub_patch

# generate hash values with: shasum -a 256 filename.pkl

RESNET50_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                  'resnet50block5-pif-paf-edge401-190625-025154-4e47f5ec.pkl')
RESNET101_MODEL = ('https://drive.switch.ch/index.php/s/9FdgUh0RAuiJdzB/download',
                   'resnet101block5-pif-paf-paf25-edge401-191012-132602-a2bf7ecd.pkl')
RESNET152_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                   'resnet152block5-pif-paf-edge401-190625-185426-3e2f28ed.pkl')
RESNEXT50_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                   'resnext50block5-pif-paf-edge401-190629-151121-24491655.pkl')
SHUFFLENETV2X1_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                        'shufflenetv2x1-pif-paf-edge401-190705-151607-d9a35d7e.pkl')
SHUFFLENETV2X2_MODEL = ('https://drive.switch.ch/index.php/s/SveBDXZBvVyl8cX/download',
                        'shufflenetv2x2-pif-paf-paf25-edge401-191010-172527-ef704f06.pkl')

LOG = logging.getLogger(__name__)


# monkey patch torch.hub
if torch_hub_patch.USE_V13:
    torch.hub.load_state_dict_from_url = torch_hub_patch.v13_load_state_dict_from_url
else:
    torch.hub.load_state_dict_from_url = torch_hub_patch.v12_load_state_dict_from_url


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, process_heads=None, cross_talk=0.0):
        super(Shell, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.process_heads = process_heads
        self.cross_talk = cross_talk

    def io_scales(self):
        return [self.base_net.input_output_scale // (2 ** getattr(h, '_quad', 0))
                for h in self.head_nets]

    def forward(self, *args):
        image_batch = args[0]

        if self.training and self.cross_talk:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk

        x = self.base_net(image_batch)
        head_outputs = [hn(x) for hn in self.head_nets]

        if self.process_heads is not None:
            head_outputs = self.process_heads(*head_outputs)

        return head_outputs


class Shell2Scale(torch.nn.Module):
    def __init__(self, base_net, head_nets, reduced_stride=3):
        super(Shell2Scale, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.reduced_stride = reduced_stride

    def io_scales(self):
        return [self.base_net.input_output_scale // (2 ** getattr(h, '_quad', 0))
                for h in self.head_nets]

    @staticmethod
    def merge_heads(original_h, reduced_h,
                    logb_component_indices,
                    stride):
        mask = reduced_h[0] > original_h[0][:, :,
                                            :stride*reduced_h[0].shape[2]:stride,
                                            :stride*reduced_h[0].shape[3]:stride]
        mask_vector = torch.stack((mask, mask), dim=2)

        for ci, (original_c, reduced_c) in enumerate(zip(original_h, reduced_h)):
            if ci == 0:
                # confidence component
                reduced_c = reduced_c * 0.5
            elif ci in logb_component_indices:
                # log(b) components
                reduced_c = torch.log(torch.exp(reduced_c) * stride)
            else:
                # vectorial and scale components
                reduced_c = reduced_c * stride

            if len(original_c.shape) == 4:
                original_c[:, :,
                           :stride*reduced_c.shape[2]:stride,
                           :stride*reduced_c.shape[3]:stride][mask] = reduced_c[mask]
            elif len(original_c.shape) == 5:
                original_c[:, :, :,
                           :stride*reduced_c.shape[3]:stride,
                           :stride*reduced_c.shape[4]:stride][mask_vector] = reduced_c[mask_vector]
            else:
                raise Exception('cannot process component with shape {}'
                                ''.format(original_c.shape))

    def forward(self, *args):
        original_input = args[0]
        original_x = self.base_net(original_input)
        original_heads = [hn(original_x) for hn in self.head_nets]

        reduced_input = original_input[:, :, ::self.reduced_stride, ::self.reduced_stride]
        reduced_x = self.base_net(reduced_input)
        reduced_heads = [hn(reduced_x) for hn in self.head_nets]

        logb_component_indices = [(2,), (3, 4)]

        for original_h, reduced_h, lci in zip(original_heads,
                                              reduced_heads,
                                              logb_component_indices):
            self.merge_heads(original_h, reduced_h, lci, self.reduced_stride)

        return original_heads


class ShellMultiScale(torch.nn.Module):
    def __init__(self, base_net, head_nets, process_heads=None, include_hflip=True):
        super(ShellMultiScale, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.pif_hflip = heads.PifHFlip(COCO_KEYPOINTS, HFLIP)
        self.paf_hflip = heads.PafHFlip(COCO_KEYPOINTS, COCO_PERSON_SKELETON, HFLIP)
        self.paf_hflip_dense = heads.PafHFlip(
            COCO_KEYPOINTS, DENSER_COCO_PERSON_CONNECTIONS, HFLIP)
        self.process_heads = process_heads
        self.include_hflip = include_hflip

    def io_scales(self):
        return [self.base_net.input_output_scale // (2 ** getattr(h, '_quad', 0))
                for h in self.head_nets]

    def forward(self, *args):
        original_input = args[0]

        head_outputs = []
        for hflip in ([False, True] if self.include_hflip else [False]):
            for reduction in [1, 1.5, 2, 3, 5]:
                if reduction == 1.5:
                    x_red = torch.ByteTensor(
                        [i % 3 != 2 for i in range(original_input.shape[3])])
                    y_red = torch.ByteTensor(
                        [i % 3 != 2 for i in range(original_input.shape[2])])
                    reduced_input = original_input[:, :, y_red, :]
                    reduced_input = reduced_input[:, :, :, x_red]
                else:
                    reduced_input = original_input[:, :, ::reduction, ::reduction]

                if hflip:
                    reduced_input = torch.flip(reduced_input, dims=[3])

                reduced_x = self.base_net(reduced_input)
                head_outputs += [hn(reduced_x) for hn in self.head_nets]

        if self.include_hflip:
            for mscale_i in range(5, 10):
                head_i = mscale_i * 3
                head_outputs[head_i] = self.pif_hflip(*head_outputs[head_i])
                head_outputs[head_i + 1] = self.paf_hflip(*head_outputs[head_i + 1])
                head_outputs[head_i + 2] = self.paf_hflip_dense(*head_outputs[head_i + 2])

        if self.process_heads is not None:
            head_outputs = self.process_heads(*head_outputs)

        return head_outputs


class Shell2Stage(torch.nn.Module):
    def __init__(self, base_net, head_nets1, head_nets2):
        super(Shell2Stage, self).__init__()

        self.base_net = base_net
        self.head_nets1 = torch.nn.ModuleList(head_nets1)
        self.head_nets2 = torch.nn.ModuleList(head_nets2)

    @property
    def head_nets(self):
        return list(self.head_nets1) + list(self.head_nets2)

    def io_scales(self):
        return (
            [self.base_net.input_output_scale[0] for _ in self.head_nets1] +
            [self.base_net.input_output_scale[1] for _ in self.head_nets2]
        )

    def forward(self, *args):
        x1, x2 = self.base_net(*args)
        h1 = [hn(x1) for hn in self.head_nets1]
        h2 = [hn(x2) for hn in self.head_nets2]
        return [h for hs in (h1, h2) for h in hs]


class ShellFork(torch.nn.Module):
    def __init__(self, base_net, head_nets1, head_nets2, head_nets3):
        super(ShellFork, self).__init__()

        self.base_net = base_net
        self.head_nets1 = torch.nn.ModuleList(head_nets1)
        self.head_nets2 = torch.nn.ModuleList(head_nets2)
        self.head_nets3 = torch.nn.ModuleList(head_nets3)

    @property
    def head_nets(self):
        return list(self.head_nets1) + list(self.head_nets2) + list(self.head_nets3)

    def io_scales(self):
        return (
            [self.base_net.input_output_scale[0] for _ in self.head_nets1] +
            [self.base_net.input_output_scale[1] for _ in self.head_nets2] +
            [self.base_net.input_output_scale[2] for _ in self.head_nets3]
        )

    def forward(self, *args):
        x1, x2, x3 = self.base_net(*args)
        h1 = [hn(x1) for hn in self.head_nets1]
        h2 = [hn(x2) for hn in self.head_nets2]
        h3 = [hn(x3) for hn in self.head_nets3]
        return [h for hs in (h1, h2, h3) for h in hs]


def factory_from_args(args):
    # configure CompositeField
    heads.CompositeField.dropout_p = args.head_dropout
    heads.CompositeField.quad = args.head_quad

    return factory(
        checkpoint=args.checkpoint,
        basenet=args.basenet,
        headnets=args.headnets,
        pretrained=args.pretrained,
        experimental=getattr(args, 'experimental_decoder', False),
        cross_talk=args.cross_talk,
        two_scale=args.two_scale,
        multi_scale=args.multi_scale,
        multi_scale_hflip=args.multi_scale_hflip,
    )


# pylint: disable=too-many-branches
def model_migration(net_cpu):
    model_defaults(net_cpu)

    for m in net_cpu.modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if not hasattr(m, 'padding_mode'):  # introduced in PyTorch 1.1.0
            m.padding_mode = 'zeros'

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    for head in net_cpu.head_nets:
        head.shortname = head.shortname.replace('PartsIntensityFields', 'pif')
        head.shortname = head.shortname.replace('PartsAssociationFields', 'paf')
        if not hasattr(head, 'dropout') or head.dropout is None:
            head.dropout = torch.nn.Dropout2d(p=0.0)
        if not hasattr(head, '_quad'):
            if hasattr(head, 'quad'):
                head._quad = head.quad  # pylint: disable=protected-access
            else:
                head._quad = 0  # pylint: disable=protected-access
        if not hasattr(head, 'scale_conv'):
            head.scale_conv = None
        if not hasattr(head, 'reg1_spread'):
            head.reg1_spread = None
        if not hasattr(head, 'reg2_spread'):
            head.reg2_spread = None
        if head.shortname == 'pif17' and getattr(head, 'scale_conv') is not None:
            head.shortname = 'pifs17'
        if head._quad == 1 and not hasattr(head, 'dequad_op'):  # pylint: disable=protected-access
            head.dequad_op = torch.nn.PixelShuffle(2)
        if not hasattr(head, 'class_convs') and hasattr(head, 'class_conv'):
            head.class_convs = torch.nn.ModuleList([head.class_conv])


def model_defaults(net_cpu):
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            m.eps = 1e-4  # tf default is 0.001

            # less momentum for variance and expectation
            m.momentum = 0.01  # tf default is 0.99


# pylint: disable=too-many-branches
def factory(
        *,
        checkpoint=None,
        basenet=None,
        headnets=('pif', 'paf'),
        pretrained=True,
        dilation=None,
        dilation_end=None,
        experimental=False,
        cross_talk=0.0,
        two_scale=False,
        multi_scale=False,
        multi_scale_hflip=True,
    ):
    if not checkpoint and basenet:
        net_cpu = factory_from_scratch(basenet, headnets, pretrained=pretrained)
        epoch = 0
    else:
        if not checkpoint:
            checkpoint = torch.utils.model_zoo.load_url(RESNET50_MODEL)
        elif checkpoint == 'resnet50':
            checkpoint = torch.utils.model_zoo.load_url(RESNET50_MODEL)
        elif checkpoint == 'resnet101':
            checkpoint = torch.utils.model_zoo.load_url(RESNET101_MODEL)
        elif checkpoint == 'resnet152':
            checkpoint = torch.utils.model_zoo.load_url(RESNET152_MODEL)
        elif checkpoint == 'resnext50':
            checkpoint = torch.utils.model_zoo.load_url(RESNEXT50_MODEL)
        elif checkpoint == 'shufflenetv2x1':
            checkpoint = torch.utils.model_zoo.load_url(SHUFFLENETV2X1_MODEL)
        elif checkpoint == 'shufflenetv2x2':
            checkpoint = torch.hub.load_state_dict_from_url(
                SHUFFLENETV2X2_MODEL[0], rename_file=SHUFFLENETV2X2_MODEL[1])
        elif checkpoint.startswith('http'):
            checkpoint = torch.utils.model_zoo.load_url(checkpoint)
        else:
            checkpoint = torch.load(checkpoint)
        net_cpu = checkpoint['model']
        epoch = checkpoint['epoch']

        # initialize for eval
        net_cpu.eval()

        # normalize for backwards compatibility
        model_migration(net_cpu)

    if experimental and not multi_scale:
        net_cpu.process_heads = heads.HeadStacks([(1, 2)])
    elif experimental and multi_scale:
        net_cpu.process_heads = heads.HeadStacks(
            [(v * 3 + 1, v * 3 + 2) for v in range(10)])
    net_cpu.cross_talk = cross_talk

    if two_scale:
        net_cpu = Shell2Scale(net_cpu.base_net, net_cpu.head_nets)

    if multi_scale:
        net_cpu = ShellMultiScale(net_cpu.base_net, net_cpu.head_nets,
                                  process_heads=net_cpu.process_heads,
                                  include_hflip=multi_scale_hflip)

    if dilation is not None:
        net_cpu.base_net.atrous0(dilation)
        # for head in net_cpu.head_nets:
        #     head.dilation = dilation
    if dilation_end is not None:
        if dilation_end == 1:
            net_cpu.base_net.atrous((1, 1))
        elif dilation_end == 2:
            net_cpu.base_net.atrous((1, 2))
        elif dilation_end == 4:
            net_cpu.base_net.atrous((2, 4))
        else:
            raise Exception
        # for head in net_cpu.head_nets:
        #     head.dilation = (dilation or 1.0) * dilation_end

    return net_cpu, epoch


# pylint: disable=too-many-return-statements
def factory_from_scratch(basename, headnames, *, pretrained=True):
    if 'resnet18' in basename:
        base_vision = torchvision.models.resnet18(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, 512, headnames)
    if 'resnet50' in basename:
        base_vision = torchvision.models.resnet50(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'resnet101' in basename:
        base_vision = torchvision.models.resnet101(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'resnet152' in basename:
        base_vision = torchvision.models.resnet152(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'resnet260' in basename:
        assert pretrained is False
        base_vision = torchvision.models.ResNet(
            torchvision.models.resnet.Bottleneck, [3, 8, 72, 3])
        return resnet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'resnext50' in basename:
        base_vision = torchvision.models.resnext50_32x4d(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'resnext101' in basename:
        base_vision = torchvision.models.resnext101_32x8d(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if basename == 'shufflenetv2x1':
        base_vision = torchvision.models.shufflenet_v2_x1_0(pretrained)
        return shufflenet_factory_from_scratch(basename, base_vision, 1024, headnames)
    if basename == 'shufflenetv2x2':
        base_vision = torchvision.models.shufflenet_v2_x2_0(pretrained)
        return shufflenet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'shufflenetv2x2w' in basename:
        base_vision = torchvision.models.ShuffleNetV2(
            [4, 8, 4], [24, 244, 488, 976, 3072],
        )
        return shufflenet_factory_from_scratch(basename, base_vision, 3072, headnames)
    # if basename == 'densenet121':
    #     basenet = basenetworks.DenseNet(torchvision.models.densenet121(pretrained), 'DenseNet121')
    # else:
    #     raise Exception('basenet not supported')

    raise Exception('unknown base network in {}'.format(basename))


def shufflenet_factory_from_scratch(basename, base_vision, out_features, headnames):
    blocks = basenetworks.ShuffleNetV2Factory(base_vision).blocks()
    basenet = basenetworks.BaseNetwork(
        torch.nn.Sequential(*blocks),
        basename,
        input_output_scale=16,
        out_features=out_features,
    )
    headnets = [heads.factory(h, basenet.out_features) for h in headnames if h != 'skeleton']
    net_cpu = Shell(basenet, headnets)
    model_defaults(net_cpu)
    return net_cpu


def resnet_factory_from_scratch(basename, base_vision, out_features, headnames):
    resnet_factory = basenetworks.ResnetBlocks(base_vision)

    # input block
    use_pool = 'pool0' in basename
    conv_stride = 2
    if 'is4' in basename:
        conv_stride = 4
    if 'is1' in basename:
        conv_stride = 1
    pool_stride = 2
    if 'pool0s4' in basename:
        pool_stride = 4

    # all blocks
    blocks = [
        resnet_factory.input_block(use_pool, conv_stride, pool_stride),
        resnet_factory.block2(),
        resnet_factory.block3(),
        resnet_factory.block4(),
    ]
    if 'block4' not in basename:
        blocks.append(resnet_factory.block5())
    else:
        out_features //= 2

    # downsample
    if 'concat' in basename:
        for b in blocks[2:]:
            resnet_factory.replace_downsample(b)

    if 'pifb' in headnames or 'pafb' in headnames:
        # TODO
        basenet = basenetworks.BaseNetwork(
            torch.nn.ModuleList([torch.nn.Sequential(*blocks[:-1]), blocks[-1]]),
            basename,
            [resnet_factory.stride(blocks[:-1]), resnet_factory.stride(blocks)],
            [out_features // 2, out_features],
        )
        head1 = [heads.factory(h, basenet.out_features[0])
                 for h in headnames if h.endswith('b')]
        head2 = [heads.factory(h, basenet.out_features[1])
                 for h in headnames if not h.endswith('b')]
        return Shell2Stage(basenet, head1, head2)

    if 'ppif' in headnames:
        # TODO
        head2 = [heads.factory(h, basenet.out_features[1])
                 for h in headnames if h == 'ppif']
        head3 = [heads.factory(h, basenet.out_features[2])
                 for h in headnames if h != 'ppif']
        return ShellFork(basenet, [], head2, head3)

    basenet = basenetworks.BaseNetwork(
        torch.nn.Sequential(*blocks),
        basename,
        input_output_scale=resnet_factory.stride(blocks),
        out_features=out_features,
    )
    headnets = [heads.factory(h, basenet.out_features) for h in headnames if h != 'skeleton']
    net_cpu = Shell(basenet, headnets)
    model_defaults(net_cpu)
    return net_cpu


def cli(parser):
    group = parser.add_argument_group('network configuration')
    group.add_argument('--checkpoint', default=None,
                       help=('Load a model from a checkpoint. '
                             'Use "resnet50", "resnet101" '
                             'or "resnet152" for pretrained OpenPifPaf models.'))
    group.add_argument('--dilation', default=None, type=int,
                       help='[never-worked] apply atrous')
    group.add_argument('--dilation-end', default=None, type=int,
                       help='[never-worked] apply atrous')
    group.add_argument('--basenet', default=None,
                       help='base network, e.g. resnet50block5')
    group.add_argument('--headnets', default=['pif', 'paf'], nargs='+',
                       help='head networks')
    group.add_argument('--no-pretrain', dest='pretrained', default=True, action='store_false',
                       help='create model without ImageNet pretraining')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--multi-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--no-multi-scale-hflip',
                       dest='multi_scale_hflip', default=True, action='store_false',
                       help='[experimental]')
    group.add_argument('--cross-talk', default=0.0, type=float,
                       help='[experimental]')

    group = parser.add_argument_group('head')
    group.add_argument('--head-dropout', default=heads.CompositeField.dropout_p, type=float,
                       help='[experimental] zeroing probability of feature in head input')
    group.add_argument('--head-quad', default=heads.CompositeField.quad, type=int,
                       help='number of times to apply quad (subpixel conv) to heads')
