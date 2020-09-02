import logging
import torch
import torchvision.models

LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, name, net, *, stride, out_features):
        super().__init__()
        self.name = name
        self.net = net
        self.stride = stride
        self.out_features = out_features
        LOG.info('%s: stride = %d, output features = %d', name, stride, out_features)

    def forward(self, *args):
        return self.net(*args)

    @classmethod
    def cli(cls, parser):
        pass

    @classmethod
    def configure(cls, args):
        pass


class ShuffleNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_shufflenetv2, out_features=2048):
        base_vision = torchvision_shufflenetv2(self.pretrained)

        net = torch.nn.Sequential(
            base_vision.conv1,
            # base_vision.maxpool,
            base_vision.stage2,
            base_vision.stage3,
            base_vision.stage4,
            base_vision.conv5,
        )
        super().__init__(name, net, stride=16, out_features=out_features)

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('ShuffleNetv2')
        assert cls.pretrained
        group.add_argument('--shufflenetv2-no-pretrain', dest='shufflenetv2_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')

    @classmethod
    def configure(cls, args):
        cls.pretrained = args.shufflenetv2_pretrained


class Resnet(BaseNetwork):
    pretrained = True
    pool0_stride = 0
    input_conv_stride = 2
    input_conv2_stride = 0
    remove_last_block = False

    def __init__(self, name, torchvision_resnet, out_features=2048):
        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32

        input_block = modules[:4]

        # input pool
        if self.pool0_stride:
            if self.pool0_stride != 2:
                input_block[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)  # pylint: disable=protected-access
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_block.pop(3)
            stride //= 2

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            assert not self.pool0_stride  # this is only intended as a replacement for maxpool
            channels = input_block[0].out_channels
            conv2 = torch.nn.Conv2d(channels, channels,
                                    kernel_size=3, stride=2, padding=1, bias=False)
            bn2 = torch.nn.BatchNorm2d(channels)
            relu2 = torch.nn.ReLU(inplace=True)
            input_block += [conv2, bn2, relu2]
            stride *= 2
            LOG.debug('replaced max pool with [3x3 conv, bn, relu] with %d channels', channels)

        # input conv
        if self.input_conv_stride != 2:
            input_block[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)  # pylint: disable=protected-access
            stride = int(stride * 2 / self.input_conv_stride)

        blocks = [
            *input_block,
            modules[4],  # block 2, no stride
            modules[5],  # block 3
            modules[6],  # block 4
            modules[7],  # block 5
        ]

        # block 5
        if self.remove_last_block:
            blocks = blocks[:-1]
            stride //= 2
            out_features //= 2

        net = torch.nn.Sequential(*blocks)
        super().__init__(name, net, stride=stride, out_features=out_features)

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('ResNet')
        assert cls.pretrained
        group.add_argument('--resnet-no-pretrain', dest='resnet_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')
        group.add_argument('--resnet-pool0-stride',
                           default=cls.pool0_stride, type=int,
                           help='stride of zero removes the pooling op')
        group.add_argument('--resnet-input-conv-stride',
                           default=cls.input_conv_stride, type=int,
                           help='stride of the input convolution')
        group.add_argument('--resnet-input-conv2-stride',
                           default=cls.input_conv2_stride, type=int,
                           help='stride of the optional 2nd input convolution')
        assert not cls.remove_last_block
        group.add_argument('--resnet-remove-last-block',
                           default=False, action='store_true',
                           help='create a network without the last block')

    @classmethod
    def configure(cls, args):
        cls.pretrained = args.resnet_pretrained
        cls.pool0_stride = args.resnet_pool0_stride
        cls.input_conv_stride = args.resnet_input_conv_stride
        cls.input_conv2_stride = args.resnet_input_conv2_stride
        cls.remove_last_block = args.resnet_remove_last_block


class InvertedResidualK(torch.nn.Module):
    """This is exactly the same as torchvision.models.shufflenet.InvertedResidual
    but with a dilation parameter."""
    def __init__(self, inp, oup, stride, *, layer_norm, dilation=1, kernel_size=3):
        super().__init__()

        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        assert dilation == 1 or kernel_size == 3
        padding = 1
        if dilation != 1:
            padding = dilation
        elif kernel_size != 3:
            padding = (kernel_size - 1) // 2

        if self.stride > 1:
            self.branch1 = torch.nn.Sequential(
                self.depthwise_conv(inp, inp,
                                    kernel_size=kernel_size, stride=self.stride,
                                    padding=padding, dilation=dilation),
                layer_norm(inp),
                torch.nn.Conv2d(inp, branch_features,
                                kernel_size=1, stride=1, padding=0, bias=False),
                layer_norm(branch_features),
                torch.nn.ReLU(inplace=True),
            )

        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(inp if (self.stride > 1) else branch_features, branch_features,
                            kernel_size=1, stride=1, padding=0, bias=False),
            layer_norm(branch_features),
            torch.nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=kernel_size, stride=self.stride,
                                padding=padding, dilation=dilation),
            layer_norm(branch_features),
            torch.nn.Conv2d(branch_features, branch_features,
                            kernel_size=1, stride=1, padding=0, bias=False),
            layer_norm(branch_features),
            torch.nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(in_f, out_f, kernel_size, stride=1, padding=0, bias=False, dilation=1):
        return torch.nn.Conv2d(in_f, out_f, kernel_size, stride, padding,
                               bias=bias, groups=in_f, dilation=dilation)

    def forward(self, *args):
        x = args[0]
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = torchvision.models.shufflenetv2.channel_shuffle(out, 2)

        return out


class ShuffleNetV2K(BaseNetwork):
    """Based on torchvision.models.ShuffleNetV2 where
    the kernel size in stages 2,3,4 is 5 instead of 3."""
    input_conv2_stride = 0
    layer_norm = None

    def __init__(self, name, stages_repeats, stages_out_channels):
        layer_norm = self.layer_norm
        if layer_norm is None:
            layer_norm = torch.nn.BatchNorm2d

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        _stage_out_channels = stages_out_channels

        stride = 16  # in the default configuration
        blocks = []
        input_channels = 3
        output_channels = _stage_out_channels[0]
        conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            layer_norm(output_channels),
            torch.nn.ReLU(inplace=True),
        )
        blocks.append(conv1)
        input_channels = output_channels

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, input_channels, 3, 2, 1, bias=False),
                layer_norm(input_channels),
                torch.nn.ReLU(inplace=True),
            )
            blocks.append(conv2)
            stride *= 2
            LOG.debug('replaced max pool with [3x3 conv, bn, relu] with %d channels',
                      input_channels)

        for repeats, output_channels in zip(
                stages_repeats, _stage_out_channels[1:]):
            seq = [InvertedResidualK(input_channels, output_channels, 2,
                                     layer_norm=layer_norm)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualK(output_channels, output_channels, 1,
                                             kernel_size=5, layer_norm=layer_norm))
            blocks.append(torch.nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = _stage_out_channels[-1]
        conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            layer_norm(output_channels),
            torch.nn.ReLU(inplace=True),
        )
        blocks.append(conv5)

        net = torch.nn.Sequential(*blocks)
        super().__init__(name, net, stride=stride, out_features=output_channels)

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('shufflenetv2k')
        group.add_argument('--shufflenetv2k-input-conv2-stride',
                           default=cls.input_conv2_stride, type=int,
                           help='stride of the optional 2nd input convolution')
        layer_norm_group = group.add_mutually_exclusive_group()
        layer_norm_group.add_argument('--shufflenetv2k-instance-norm',
                                      default=False, action='store_true')
        layer_norm_group.add_argument('--shufflenetv2k-group-norm',
                                      default=False, action='store_true')

    @classmethod
    def configure(cls, args):
        cls.input_conv2_stride = args.shufflenetv2k_input_conv2_stride
        # layer norms
        if args.shufflenetv2k_instance_norm:
            cls.layer_norm = lambda x: torch.nn.InstanceNorm2d(
                x, eps=1e-4, momentum=0.01, affine=True, track_running_stats=True)
        if args.shufflenetv2k_group_norm:
            cls.layer_norm = lambda x: torch.nn.GroupNorm(32 if x > 100 else 4, x, eps=1e-4)
