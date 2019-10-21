import logging
import torch

LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, net, shortname, input_output_scale, out_features):
        super(BaseNetwork, self).__init__()

        self.net = net
        self.shortname = shortname
        self.input_output_scale = input_output_scale
        self.out_features = out_features

        # print(list(net.children()))
        LOG.info('stide = %d', self.input_output_scale)
        LOG.info('output features = %d', self.out_features)

    def forward(self, *args):
        return self.net(*args)


class ShuffleNetV2Factory(object):
    def __init__(self, torchvision_shufflenetv2):
        self.torchvision_shufflenetv2 = torchvision_shufflenetv2

    def blocks(self):
        return [
            self.torchvision_shufflenetv2.conv1,
            # self.torchvision_shufflenetv2.maxpool,
            self.torchvision_shufflenetv2.stage2,
            self.torchvision_shufflenetv2.stage3,
            self.torchvision_shufflenetv2.stage4,
            self.torchvision_shufflenetv2.conv5,
        ]


class DownsampleCat(torch.nn.Module):
    def __init__(self):
        super(DownsampleCat, self).__init__()
        self.pad = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.0)

    def forward(self, x):  # pylint: disable=arguments-differ
        p = self.pad(x)
        o = torch.cat((p[:, :, :-1:2, :-1:2], p[:, :, 1::2, 1::2]), dim=1)
        return o


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

    @staticmethod
    def stride(block):
        """Compute the output stride of a block.

        Assume that convolutions are in serious with pools; only one
        convolutions with non-unit stride.
        """
        if isinstance(block, list):
            stride = 1
            for b in block:
                stride *= ResnetBlocks.stride(b)
            return stride

        conv_stride = max(m.stride[0]
                          for m in block.modules()
                          if isinstance(m, torch.nn.Conv2d))

        pool_stride = 1
        pools = [m for m in block.modules() if isinstance(m, torch.nn.MaxPool2d)]
        if pools:
            for p in pools:
                pool_stride *= p.stride

        return conv_stride * pool_stride

    @staticmethod
    def replace_downsample(block):
        print('!!!!!!!!!!')
        first_bottleneck = block[0]
        print(first_bottleneck.downsample)
        first_bottleneck.downsample = DownsampleCat()
        print(first_bottleneck)

    def block2(self):
        return self.modules[4]

    def block3(self):
        return self.modules[5]

    def block4(self):
        return self.modules[6]

    def block5(self):
        return self.modules[7]
