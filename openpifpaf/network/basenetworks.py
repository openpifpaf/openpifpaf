import logging
import torch
import torchvision.models

LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, net, shortname, stride, out_features):
        super(BaseNetwork, self).__init__()

        self.net = net
        self.shortname = shortname
        self.stride = stride
        self.out_features = out_features

        # print(list(net.children()))
        LOG.info('stride = %d', self.stride)
        LOG.info('output features = %d', self.out_features)

    def forward(self, *args):
        return self.net(*args)


class ResnetBlocks(object):
    def __init__(self, resnet):
        self.modules = list(resnet.children())
        LOG.debug('modules = %s', self.modules)

    def input_block(self, use_pool=False, conv_stride=2, pool_stride=2):
        modules = self.modules[:4]

        if not use_pool:
            modules.pop(3)
        else:
            if pool_stride != 2:
                modules[3].stride = torch.nn.modules.utils._pair(pool_stride)  # pylint: disable=protected-access

        if conv_stride != 2:
            modules[0].stride = torch.nn.modules.utils._pair(conv_stride)  # pylint: disable=protected-access

        return torch.nn.Sequential(*modules)

    def block2(self):
        return self.modules[4]

    def block3(self):
        return self.modules[5]

    def block4(self):
        return self.modules[6]

    def block5(self):
        return self.modules[7]


class InvertedResidualK(torch.nn.Module):
    """This is exactly the same as torchvision.models.shufflenet.InvertedResidual
    but with a dilation parameter."""
    def __init__(self, inp, oup, stride, *, layer_norm, dilation=1, kernel_size=3):
        super(InvertedResidualK, self).__init__()

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


class ShuffleNetV2K(torch.nn.Module):
    """Based on torchvision.models.ShuffleNetV2 where
    the kernel size in stages 2,3,4 is 5 instead of 3."""
    def __init__(self, stages_repeats, stages_out_channels, *, layer_norm=None):
        super(ShuffleNetV2K, self).__init__()
        if layer_norm is None:
            layer_norm = torch.nn.BatchNorm2d

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            layer_norm(output_channels),
            torch.nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidualK(input_channels, output_channels, 2,
                                     layer_norm=layer_norm)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualK(output_channels, output_channels, 1,
                                             kernel_size=5, layer_norm=layer_norm))
            setattr(self, name, torch.nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            layer_norm(output_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, *args):
        x = args[0]
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x
