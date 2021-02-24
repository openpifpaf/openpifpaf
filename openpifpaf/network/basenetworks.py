import argparse
import logging
import torch
import torchvision.models

LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network.

    :param name: a short name for the base network, e.g. resnet50
    :param stride: total stride from input to output
    :param out_features: number of output features
    """

    def __init__(self, name: str, *, stride: int, out_features: int):
        super().__init__()
        self.name = name
        self.stride = stride
        self.out_features = out_features
        LOG.info('%s: stride = %d, output features = %d', name, stride, out_features)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, *args):
        raise NotImplementedError


class ShuffleNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_shufflenetv2, out_features=2048):
        super().__init__(name, stride=16, out_features=out_features)

        base_vision = torchvision_shufflenetv2(self.pretrained)
        self.conv1 = base_vision.conv1
        # base_vision.maxpool
        self.stage2 = base_vision.stage2
        self.stage3 = base_vision.stage3
        self.stage4 = base_vision.stage4
        self.conv5 = base_vision.conv5

    def forward(self, *args):
        x = args[0]
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('ShuffleNetv2')
        assert cls.pretrained
        group.add_argument('--shufflenetv2-no-pretrain', dest='shufflenetv2_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.shufflenetv2_pretrained


class Resnet(BaseNetwork):
    pretrained = True
    pool0_stride = 0
    input_conv_stride = 2
    input_conv2_stride = 0
    remove_last_block = False
    block5_dilation = 1

    def __init__(self, name, torchvision_resnet, out_features=2048):
        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32

        input_modules = modules[:4]

        # input pool
        if self.pool0_stride:
            if self.pool0_stride != 2:
                # pylint: disable=protected-access
                input_modules[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_modules.pop(3)
            stride //= 2

        # input conv
        if self.input_conv_stride != 2:
            # pylint: disable=protected-access
            input_modules[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)
            stride = int(stride * 2 / self.input_conv_stride)

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            assert not self.pool0_stride  # this is only intended as a replacement for maxpool
            channels = input_modules[0].out_channels
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(inplace=True),
            )
            input_modules.append(conv2)
            stride *= 2
            LOG.debug('replaced max pool with [3x3 conv, bn, relu] with %d channels', channels)

        # block 5
        block5 = modules[7]
        if self.remove_last_block:
            block5 = None
            stride //= 2
            out_features //= 2

        if self.block5_dilation != 1:
            stride //= 2
            for m in block5.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue

                # also must be changed for the skip-conv that has kernel=1
                m.stride = torch.nn.modules.utils._pair(1)

                if m.kernel_size[0] == 1:
                    continue

                m.dilation = torch.nn.modules.utils._pair(self.block5_dilation)
                padding = (m.kernel_size[0] - 1) // 2 * self.block5_dilation
                m.padding = torch.nn.modules.utils._pair(padding)

        super().__init__(name, stride=stride, out_features=out_features)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = block5

    def forward(self, *args):
        x = args[0]
        x = self.input_block(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
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
        group.add_argument('--resnet-block5-dilation',
                           default=cls.block5_dilation, type=int,
                           help='use dilated convs in block5')
        assert not cls.remove_last_block
        group.add_argument('--resnet-remove-last-block',
                           default=False, action='store_true',
                           help='create a network without the last block')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.resnet_pretrained
        cls.pool0_stride = args.resnet_pool0_stride
        cls.input_conv_stride = args.resnet_input_conv_stride
        cls.input_conv2_stride = args.resnet_input_conv2_stride
        cls.block5_dilation = args.resnet_block5_dilation
        cls.remove_last_block = args.resnet_remove_last_block


class InvertedResidualK(torch.nn.Module):
    """Based on torchvision.models.shufflenet.InvertedResidual."""

    def __init__(self, inp, oup, first_in_stage, *,
                 stride=1, layer_norm, non_linearity, dilation=1, kernel_size=3):
        super().__init__()
        assert (stride != 1 or dilation != 1 or inp != oup) or not first_in_stage
        LOG.debug('InvResK: %d %d %s, stride=%d, dilation=%d',
                  inp, oup, first_in_stage, stride, dilation)

        self.first_in_stage = first_in_stage
        branch_features = oup // 2
        padding = (kernel_size - 1) // 2 * dilation

        if self.first_in_stage:
            self.branch1 = torch.nn.Sequential(
                self.depthwise_conv(inp, inp,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation),
                layer_norm(inp),
                torch.nn.Conv2d(inp, branch_features,
                                kernel_size=1, stride=1, padding=0, bias=False),
                layer_norm(branch_features),
                non_linearity(),
            )

        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(inp if first_in_stage else branch_features, branch_features,
                            kernel_size=1, stride=1, padding=0, bias=False),
            layer_norm(branch_features),
            non_linearity(),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation),
            layer_norm(branch_features),
            torch.nn.Conv2d(branch_features, branch_features,
                            kernel_size=1, stride=1, padding=0, bias=False),
            layer_norm(branch_features),
            non_linearity(),
        )

    @staticmethod
    def depthwise_conv(in_f, out_f, kernel_size, stride=1, padding=0, bias=False, dilation=1):
        return torch.nn.Conv2d(in_f, out_f, kernel_size, stride, padding,
                               bias=bias, groups=in_f, dilation=dilation)

    def forward(self, *args):
        x = args[0]
        if not self.first_in_stage:
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
    input_conv2_outchannels = None
    layer_norm = None
    stage4_dilation = 1
    kernel_width = 5
    conv5_as_stage = False
    non_linearity = None

    def __init__(self, name, stages_repeats, stages_out_channels):
        layer_norm = ShuffleNetV2K.layer_norm
        if layer_norm is None:
            layer_norm = torch.nn.BatchNorm2d
        non_linearity = ShuffleNetV2K.non_linearity
        if non_linearity is None:
            non_linearity = lambda: torch.nn.ReLU(inplace=True)

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        _stage_out_channels = stages_out_channels

        stride = 16  # in the default configuration
        input_modules = []
        input_channels = 3
        output_channels = _stage_out_channels[0]
        conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            layer_norm(output_channels),
            non_linearity(),
        )
        input_modules.append(conv1)
        input_channels = output_channels

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            output_channels = self.input_conv2_outchannels or input_channels
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                layer_norm(output_channels),
                non_linearity(),
            )
            input_modules.append(conv2)
            stride *= 2
            input_channels = output_channels
            LOG.debug('replaced max pool with [3x3 conv, bn, relu]')

        stages = []
        for repeats, output_channels, dilation in zip(
                stages_repeats, _stage_out_channels[1:], [1, 1, self.stage4_dilation]):
            stage_stride = 2 if dilation == 1 else 1
            stride = int(stride * stage_stride / 2)
            seq = [InvertedResidualK(input_channels, output_channels, True,
                                     kernel_size=self.kernel_width,
                                     layer_norm=layer_norm,
                                     non_linearity=non_linearity,
                                     dilation=dilation,
                                     stride=stage_stride)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualK(output_channels, output_channels, False,
                                             kernel_size=self.kernel_width,
                                             layer_norm=layer_norm,
                                             non_linearity=non_linearity,
                                             dilation=dilation))
            stages.append(torch.nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = _stage_out_channels[-1]
        if self.conv5_as_stage:
            # Two stages are about the same number of parameters as one
            # convolution.
            # Conv: 1392*1392
            # Two Stages: 4 * 696*696 + 2 * 5^2*696
            use_first_in_stage = input_channels != output_channels
            conv5 = torch.nn.Sequential(
                InvertedResidualK(input_channels, output_channels, use_first_in_stage,
                                  kernel_size=self.kernel_width,
                                  layer_norm=layer_norm,
                                  non_linearity=non_linearity,
                                  dilation=self.stage4_dilation),
                InvertedResidualK(output_channels, output_channels, False,
                                  kernel_size=self.kernel_width,
                                  layer_norm=layer_norm,
                                  non_linearity=non_linearity,
                                  dilation=self.stage4_dilation),
            )
        else:
            conv5 = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                layer_norm(output_channels),
                non_linearity(),
            )

        super().__init__(name, stride=stride, out_features=output_channels)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.stage2 = stages[0]
        self.stage3 = stages[1]
        self.stage4 = stages[2]
        self.conv5 = conv5

    def forward(self, *args):
        x = args[0]
        x = self.input_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('shufflenetv2k')
        group.add_argument('--shufflenetv2k-input-conv2-stride',
                           default=cls.input_conv2_stride, type=int,
                           help='stride of the optional 2nd input convolution')
        group.add_argument('--shufflenetv2k-input-conv2-outchannels',
                           default=cls.input_conv2_outchannels, type=int,
                           help='out channels of the optional 2nd input convolution')
        group.add_argument('--shufflenetv2k-stage4-dilation',
                           default=cls.stage4_dilation, type=int,
                           help='dilation factor of stage 4')
        group.add_argument('--shufflenetv2k-kernel',
                           default=cls.kernel_width, type=int,
                           help='kernel width')
        assert not cls.conv5_as_stage
        group.add_argument('--shufflenetv2k-conv5-as-stage',
                           default=False, action='store_true')

        layer_norm_group = group.add_mutually_exclusive_group()
        layer_norm_group.add_argument('--shufflenetv2k-instance-norm',
                                      default=False, action='store_true')
        layer_norm_group.add_argument('--shufflenetv2k-group-norm',
                                      default=False, action='store_true')

        non_linearity_group = group.add_mutually_exclusive_group()
        non_linearity_group.add_argument('--shufflenetv2k-leaky-relu',
                                         default=False, action='store_true')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.input_conv2_stride = args.shufflenetv2k_input_conv2_stride
        cls.input_conv2_outchannels = args.shufflenetv2k_input_conv2_outchannels
        cls.stage4_dilation = args.shufflenetv2k_stage4_dilation
        cls.kernel_width = args.shufflenetv2k_kernel
        cls.conv5_as_stage = args.shufflenetv2k_conv5_as_stage

        # layer norms
        if args.shufflenetv2k_instance_norm:
            cls.layer_norm = lambda x: torch.nn.InstanceNorm2d(
                x, momentum=0.01, affine=True, track_running_stats=True)
        if args.shufflenetv2k_group_norm:
            cls.layer_norm = lambda x: torch.nn.GroupNorm(
                (32 if x % 32 == 0 else 29) if x > 100 else 4, x)

        # non-linearities
        if args.shufflenetv2k_leaky_relu:
            cls.non_linearity = lambda: torch.nn.LeakyReLU(inplace=True)


class MobileNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_mobilenetv2, out_features=1280):
        super().__init__(name, stride=32, out_features=out_features)
        base_vision = torchvision_mobilenetv2(self.pretrained)
        self.backbone = list(base_vision.children())[0]  # remove output classifier

    def forward(self, *args):
        x = args[0]
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('MobileNetV2')
        assert cls.pretrained
        group.add_argument('--mobilenetv2-no-pretrain', dest='mobilenetv2_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.mobilenetv2_pretrained


class SqueezeNet(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_squeezenet, out_features=512):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_squeezenet(self.pretrained)

        for m in base_vision.modules():
            # adjust padding on all maxpool layers
            if isinstance(m, (torch.nn.MaxPool2d,)) and m.padding != 1:
                LOG.debug('adjusting maxpool2d padding to 1 from padding=%d, kernel=%d, stride=%d',
                          m.padding, m.kernel_size, m.stride)
                m.padding = 1

            # adjust padding on some conv2d (only the first one)
            if isinstance(m, (torch.nn.Conv2d,)):
                target_padding = (m.kernel_size[0] - 1) // 2
                if m.padding[0] != target_padding:
                    LOG.debug('adjusting conv2d padding to %d (kernel=%d, padding=%d)',
                              target_padding, m.kernel_size, m.padding)
                    m.padding = torch.nn.modules.utils._pair(target_padding)  # pylint: disable=protected-access

        self.backbone = list(base_vision.children())[0]  # remove output classifier

    def forward(self, *args):
        x = args[0]
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('SqueezeNet')
        assert cls.pretrained
        group.add_argument('--squeezenet-no-pretrain', dest='squeezenet_pretrained',
                           default=True, action='store_false',
                           help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.squeezenet_pretrained
