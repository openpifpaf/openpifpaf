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
        LOG.info('stride = %d', self.input_output_scale)
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
