import copy
import torch


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, net, shortname, input_output_scale, out_features):
        super(BaseNetwork, self).__init__()

        self.net = net
        self.shortname = shortname
        self.input_output_scale = input_output_scale
        self.out_features = out_features
        self.topology = 'linear'

        # print(list(net.children()))
        print('input output scale', self.input_output_scale)
        print('output features', self.out_features)

    def forward(self, image):  # pylint: disable=arguments-differ
        if isinstance(self.net, torch.nn.ModuleList):
            if self.topology == 'linear':
                intermediate = image
                outputs = []
                for n in self.net:
                    intermediate = n(intermediate)
                    outputs.append(intermediate)

                return outputs

            if self.topology == 'fork':
                intermediate = self.net[0](image)
                return intermediate, self.net[1](intermediate), self.net[2](intermediate)

        return self.net(image)


class ResnetC4(BaseNetwork):
    """Resnet capped after stage4. Default is a Resnet50.

    Spatial resolution of output is input resolution divided by 16.
    Has an option to keep stage5.
    """

    def __init__(self, resnet, shortname=None, remove_pool0=True,
                 input_stride=2, pool0_stride=2, block5=False,
                 twostage=False, fork=False):
        # print('===============')
        # print(list(resnet.children()))

        if not block5:
            # remove the linear, avgpool2d and stage5
            stump_modules = list(resnet.children())[:-3]
            input_output_scale = 16
            out_features = 1024
        else:
            # remove linear and avgpool2d
            stump_modules = list(resnet.children())[:-2]
            input_output_scale = 32
            out_features = 2048

        if remove_pool0:
            stump_modules.pop(3)
            input_output_scale /= 2
        else:
            if pool0_stride != 2:
                stump_modules[3].stride = torch.nn.modules.utils._pair(pool0_stride)  # pylint: disable=protected-access
                input_output_scale *= pool0_stride / 2

        if input_stride != 2:
            stump_modules[0].stride = torch.nn.modules.utils._pair(input_stride)  # pylint: disable=protected-access
            input_output_scale *= input_stride / 2

        if twostage:
            stump = torch.nn.ModuleList([
                torch.nn.Sequential(*stump_modules[:-1]),
                torch.nn.Sequential(*stump_modules[-1:]),
            ])
        elif fork:
            stump = torch.nn.ModuleList([
                torch.nn.Sequential(*stump_modules[:-1]),
                torch.nn.Sequential(*stump_modules[-1:]),
                copy.deepcopy(torch.nn.Sequential(*stump_modules[-1:])),
            ])
        else:
            stump = torch.nn.Sequential(*stump_modules)

        shortname = shortname or resnet.__class__.__name__
        super(ResnetC4, self).__init__(stump, shortname, input_output_scale, out_features)
        if fork:
            self.topology = 'fork'

    def atrous0(self, dilation):
        convs = [m for m in self.net.modules() if isinstance(m, torch.nn.Conv2d)]
        first_conv = convs[0]

        print('before atrous', list(self.net.children()))
        print('model: stride = {}, dilation = {}, input_output = {}'
              ''.format(first_conv.stride, first_conv.dilation, self.input_output_scale))

        original_stride = first_conv.stride[0]
        first_conv.stride = torch.nn.modules.utils._pair(original_stride // dilation)  # pylint: disable=protected-access
        first_conv.dilation = torch.nn.modules.utils._pair(dilation)  # pylint: disable=protected-access
        padding = (first_conv.kernel_size[0] - 1) // 2 * first_conv.dilation[0]
        first_conv.padding = torch.nn.modules.utils._pair(padding)  # pylint: disable=protected-access

        for conv in convs[1:]:
            if conv.kernel_size[0] > 1:
                conv.dilation = torch.nn.modules.utils._pair(dilation)  # pylint: disable=protected-access

                padding = (conv.kernel_size[0] - 1) // 2 * conv.dilation[0]
                conv.padding = torch.nn.modules.utils._pair(padding)  # pylint: disable=protected-access

        self.input_output_scale /= dilation
        print('after atrous', list(self.net.children()))
        print('atrous modification: stride = {}, dilation = {}, input_output = {}'
              ''.format(first_conv.stride, first_conv.dilation, self.input_output_scale))

    def atrous(self, dilations):
        """Apply atrous."""
        if isinstance(self.net, tuple):
            children = list(self.net[0].children()) + list(self.net[1].children())
        else:
            children = list(self.net.children())

        layer3, layer4 = children[-2:]
        print('before layer 3', layer3)
        print('before layer 4', layer4)

        prev_dilations = [1] + list(dilations[:-1])
        for prev_dilation, dilation, layer in zip(prev_dilations, dilations, (layer3, layer4)):
            if dilation == 1:
                continue

            convs = [m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)]
            layer_stride = max(c.stride[0] for c in convs)
            self.input_output_scale /= layer_stride

            for conv in convs:
                if dilation != prev_dilation:
                    conv.stride = torch.nn.modules.utils._pair(1)  # pylint: disable=protected-access
                if conv.kernel_size[0] > 1:
                    conv.dilation = torch.nn.modules.utils._pair(dilation)  # pylint: disable=protected-access

                    padding = (conv.kernel_size[0] - 1) // 2 * dilation
                    conv.padding = torch.nn.modules.utils._pair(padding)  # pylint: disable=protected-access

        print('after atrous layer 3', layer3)
        print('after atrous layer 4', layer4)


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
        # print('===============')
        # print(self.modules)

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
    def dilation(block, dilation, stride=1):
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)]

        for conv in convs:
            if conv.kernel_size[0] == 1:
                continue

            conv.dilation = torch.nn.modules.utils._pair(dilation)  # pylint: disable=protected-access

            padding = (conv.kernel_size[0] - 1) // 2 * dilation
            conv.padding = torch.nn.modules.utils._pair(padding)  # pylint: disable=protected-access

        # TODO: check these are the right convolutions to adjust
        for conv in convs[:2]:
            conv.stride = torch.nn.modules.utils._pair(stride)  # pylint: disable=protected-access

        return block

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

    @staticmethod
    def out_channels(block):
        """For blocks 2-5."""
        last_conv = list(block.modules())[-3]
        return last_conv.out_channels

    def block2(self):
        return self.modules[4]

    def block3(self):
        return self.modules[5]

    def block4(self):
        return self.modules[6]

    def block5(self):
        return self.modules[7]


class DenseNet(BaseNetwork):
    """DenseNet. Default is a densenet121.

    Spatial resolution of output is input resolution divided by 16.
    """

    def __init__(self, densenet, shortname=None, remove_pool0=True, adjust_input_stride=False):
        # print('===============')
        # print(list(densenet.children()))
        input_output_scale = 32

        # remove the last linear layer, and maxpool0 at the beginning
        stump_modules = list(list(densenet.children())[0].children())[:-1]
        if remove_pool0:
            stump_modules.pop(3)
            input_output_scale /= 2
        if adjust_input_stride:
            stump_modules[0].stride = torch.nn.modules.utils._pair(1)  # pylint: disable=protected-access
            input_output_scale /= 2
        stump = torch.nn.Sequential(*stump_modules)

        shortname = shortname or densenet.__class__.__name__
        super(DenseNet, self).__init__(stump, shortname, input_output_scale)
