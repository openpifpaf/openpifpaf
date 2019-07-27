import logging
import torch
import torchvision

from . import basenetworks, heads

# generate hash values with: shasum -a 256 filename.pkl

RESNET50_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                  'resnet50block5-pif-paf-edge401-190625-025154-4e47f5ec.pkl')
RESNET101_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                   'resnet101block5-pif-paf-edge401-190629-151620-b2db8c7e.pkl')
RESNET152_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                   'resnet152block5-pif-paf-edge401-190625-185426-3e2f28ed.pkl')
RESNEXT50_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                   'resnext50block5-pif-paf-edge401-190629-151121-24491655.pkl')
SHUFFLENETV2X1_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                        'shufflenetv2x1-pif-paf-edge401-190705-151607-d9a35d7e.pkl')
SHUFFLENETV2X2_MODEL = ('https://storage.googleapis.com/openpifpaf-pretrained/v0.8.0/'
                        'shufflenetv2x2-pif-paf-edge401-190705-151618-f8da8c15.pkl')

LOG = logging.getLogger(__name__)


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets):
        super(Shell, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)

    def io_scales(self):
        return [self.base_net.input_output_scale // (2 ** getattr(h, '_quad', 0))
                for h in self.head_nets]

    def forward(self, *args):
        x = self.base_net(*args)
        return [hn(x) for hn in self.head_nets]


class Shell2Scale(torch.nn.Module):
    def __init__(self, base_net, head_nets):
        super(Shell2Scale, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)

    def io_scales(self):
        return [self.base_net.input_output_scale // (2 ** getattr(h, '_quad', 0))
                for h in self.head_nets]

    @staticmethod
    def merge_heads(original_h, reduced_h, logb_component_indices):
        mask = reduced_h[0] > original_h[0][:, :,
                                            :4*reduced_h[0].shape[2]:4,
                                            :4*reduced_h[0].shape[3]:4]
        mask_vector = torch.stack((mask, mask), dim=2)

        for ci, (original_c, reduced_c) in enumerate(zip(original_h, reduced_h)):
            if ci == 0:
                # confidence component
                reduced_c = reduced_c * 0.3
            elif ci in logb_component_indices:
                # log(b) components
                reduced_c = torch.log(torch.exp(reduced_c) * 4.0)
            else:
                # vectorial and scale components
                reduced_c = reduced_c * 4.0

            if len(original_c.shape) == 4:
                original_c[:, :,
                           :4*reduced_c.shape[2]:4,
                           :4*reduced_c.shape[3]:4][mask] = reduced_c[mask]
            elif len(original_c.shape) == 5:
                original_c[:, :, :,
                           :4*reduced_c.shape[3]:4,
                           :4*reduced_c.shape[4]:4][mask_vector] = reduced_c[mask_vector]
            else:
                raise Exception('cannot process component with shape {}'
                                ''.format(original_c.shape))

    def forward(self, *args):
        original_input = args[0]
        original_x = self.base_net(original_input)
        original_heads = [hn(original_x) for hn in self.head_nets]

        reduced_input = original_input[:, :, ::4, ::4]
        reduced_x = self.base_net(reduced_input)
        reduced_heads = [hn(reduced_x) for hn in self.head_nets]

        logb_component_indices = [(2,), (3, 4)]

        for original_h, reduced_h, lci in zip(original_heads,
                                              reduced_heads,
                                              logb_component_indices):
            self.merge_heads(original_h, reduced_h, lci)

        return original_heads


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
    for head in (heads.HEADS or heads.Head.__subclasses__()):
        head.apply_args(args)

    return factory(checkpoint=args.checkpoint,
                   basenet=args.basenet,
                   headnets=args.headnets,
                   pretrained=args.pretrained,
                   two_scale=args.two_scale)


# pylint: disable=too-many-branches
def model_migration(net_cpu):
    model_defaults(net_cpu)

    for m in net_cpu.modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if not hasattr(m, 'padding_mode'):  # introduced in PyTorch 1.1.0
            m.padding_mode = 'zeros'

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
def factory(*,
            checkpoint=None,
            basenet=None,
            headnets=('pif', 'paf'),
            pretrained=True,
            dilation=None,
            dilation_end=None,
            two_scale=False):
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
            checkpoint = torch.utils.model_zoo.load_url(SHUFFLENETV2X2_MODEL)
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

    if two_scale:
        net_cpu = Shell2Scale(net_cpu.base_net, net_cpu.head_nets)

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


def create_headnet(name, n_features):
    for head in (heads.HEADS or heads.Head.__subclasses__()):
        LOG.debug('checking head %s matches %s', head.__name__, name)
        if not head.match(name):
            continue
        LOG.info('selected head %s for %s', head.__name__, name)
        return head(name, n_features)

    raise Exception('unknown head to create a head network: {}'.format(name))


# pylint: disable=too-many-return-statements
def factory_from_scratch(basename, headnames, *, pretrained=True):
    if 'resnet50' in basename:
        base_vision = torchvision.models.resnet50(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, headnames)
    if 'resnet101' in basename:
        base_vision = torchvision.models.resnet101(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, headnames)
    if 'resnet152' in basename:
        base_vision = torchvision.models.resnet152(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, headnames)
    if 'resnet260' in basename:
        assert pretrained is False
        base_vision = torchvision.models.ResNet(
            torchvision.models.resnet.Bottleneck, [3, 8, 72, 3])
        return resnet_factory_from_scratch(basename, base_vision, headnames)
    if 'resnext50' in basename:
        base_vision = torchvision.models.resnext50_32x4d(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, headnames)
    if 'resnext101' in basename:
        base_vision = torchvision.models.resnext101_32x8d(pretrained)
        return resnet_factory_from_scratch(basename, base_vision, headnames)
    if basename == 'shufflenetv2x1':
        base_vision = torchvision.models.shufflenet_v2_x1_0(pretrained)
        return shufflenet_factory_from_scratch(basename, base_vision, 1024, headnames)
    if basename == 'shufflenetv2x2':
        base_vision = torchvision.models.shufflenet_v2_x2_0(pretrained)
        return shufflenet_factory_from_scratch(basename, base_vision, 2048, headnames)
    if 'shufflenetv2x2w' in basename:
        base_vision = torchvision.models._shufflenetv2(  # pylint: disable=protected-access
            'shufflenetv2_x2.0_w', pretrained, False,
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
    headnets = [create_headnet(h, basenet.out_features) for h in headnames if h != 'skeleton']
    net_cpu = Shell(basenet, headnets)
    model_defaults(net_cpu)
    return net_cpu


def resnet_factory_from_scratch(basename, base_vision, headnames):
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
    if 'block5' in basename:
        blocks.append(resnet_factory.block5())

    # downsample
    if 'concat' in basename:
        for b in blocks[2:]:
            resnet_factory.replace_downsample(b)

    if 'pifb' in headnames or 'pafb' in headnames:
        basenet = basenetworks.BaseNetwork(
            torch.nn.ModuleList([torch.nn.Sequential(*blocks[:-1]), blocks[-1]]),
            basename,
            [resnet_factory.stride(blocks[:-1]), resnet_factory.stride(blocks)],
            [resnet_factory.out_channels(blocks[-2]), resnet_factory.out_channels(blocks[-1])],
        )
        head1 = [create_headnet(h, basenet.out_features[0])
                 for h in headnames if h.endswith('b')]
        head2 = [create_headnet(h, basenet.out_features[1])
                 for h in headnames if not h.endswith('b')]
        return Shell2Stage(basenet, head1, head2)

    if 'ppif' in headnames:
        # TODO
        head2 = [create_headnet(h, basenet.out_features[1])
                 for h in headnames if h == 'ppif']
        head3 = [create_headnet(h, basenet.out_features[2])
                 for h in headnames if h != 'ppif']
        return ShellFork(basenet, [], head2, head3)

    basenet = basenetworks.BaseNetwork(
        torch.nn.Sequential(*blocks),
        basename,
        resnet_factory.stride(blocks),
        resnet_factory.out_channels(blocks[-1]),
    )
    headnets = [create_headnet(h, basenet.out_features) for h in headnames if h != 'skeleton']
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
                       help='apply atrous')
    group.add_argument('--dilation-end', default=None, type=int,
                       help='apply atrous')
    group.add_argument('--basenet', default=None,
                       help='base network, e.g. resnet50block5')
    group.add_argument('--headnets', default=['pif', 'paf'], nargs='+',
                       help='head networks')
    group.add_argument('--no-pretrain', dest='pretrained', default=True, action='store_false',
                       help='create model without ImageNet pretraining')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='two scale')

    for head in (heads.HEADS or heads.Head.__subclasses__()):
        head.cli(parser)
