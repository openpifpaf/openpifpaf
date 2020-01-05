import itertools
import logging
import torch
import torchvision

LOG = logging.getLogger(__name__)


def czip(*args):
    for a in args[1:]:
        assert len(args[0]) == len(a)
    return zip(*args)


def reversed_czip(*args):
    for a in args[1:]:
        assert len(args[0]) == len(a)
    return zip(*[reversed(a) for a in args])


class PumpAndDump(torch.nn.Module):
    n_layers = 2
    stack_size = 2
    n_features = 512

    def __init__(self, in_features, *, block_factory, lateral_factory):
        super().__init__()
        LOG.info('layers=%d, stack=%d', self.n_layers, self.stack_size)

        self.columns = torch.nn.ModuleList([
            PumpAndDumpColumn(in_features if s == 0 else self.n_features,
                              self.n_features,
                              self.n_layers,
                              block_factory=block_factory,
                              lateral_factory=lateral_factory)
            for s in range(self.stack_size)
        ] + [
            ConcatenateColumn(self.n_features, self.n_layers,
                              lateral_factory=lateral_factory)
        ])

        self.out_lateral = lateral_factory(
            self.n_features * (self.n_layers + 1),
            self.n_features * (self.n_layers + 1),
        )

    def forward(self, *args):
        x = [args[0]] + [
            args[0][:, :, 0::2**(l + 1), 0::2**(l + 1)]
            for l, _ in enumerate(self.columns[0].bottlenecks)
        ]

        for column in self.columns:
            x = column(x)

        return self.out_lateral(x[0])

    @staticmethod
    def create_bottleneck(in_features):
        return torchvision.models.resnet.Bottleneck(
            in_features, in_features // 4,
            stride=2,
            downsample=torch.nn.Sequential(
                torchvision.models.resnet.conv1x1(in_features, in_features, stride=2),
                torch.nn.BatchNorm2d(in_features),
                torch.nn.ReLU(),
            ),
        )

    @staticmethod
    def create_lateral(in_features, out_features):
        if in_features != out_features:
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
            )

        return torchvision.models.resnet.Bottleneck(
            in_features, out_features // 4,
        )

    @staticmethod
    def create_invertedresidual(in_features):
        return torchvision.models.shufflenetv2.InvertedResidual(
            in_features, in_features, stride=2)

    @staticmethod
    def create_lateral_invertedresidual(in_features, out_features):
        if in_features != out_features:
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.ReLU(),
            )

        return torchvision.models.shufflenetv2.InvertedResidual(
            in_features, out_features, stride=1)


class ConcatenateColumn(torch.nn.Module):
    def __init__(self, pyramid_features, n_layers, *, lateral_factory):
        super().__init__()

        self.upsample = UpsampleWithClip(scale_factor=2, mode='nearest')

        self.lateral = torch.nn.ModuleList([
            lateral_factory(pyramid_features, pyramid_features)
            for _ in range(n_layers + 1)
        ])

    def forward(self, *args):
        inputs = args[0]

        out = [self.lateral[-1](inputs[-1])]
        for input_, lateral in reversed_czip(inputs[:-1], self.lateral[:-1]):
            dumped = self.upsample(out[0])
            out.insert(
                0,
                torch.cat((dumped, lateral(input_)), dim=1)
            )

        return out


class UpsampleWithClip(torch.nn.Module):
    """Upsample with last row and column clip."""

    def __init__(self, **kwargs):
        super().__init__()
        self.upsample = torch.nn.Upsample(**kwargs)

    def forward(self, *args):
        return self.upsample(args[0])[:, :, :-1, :-1]


class SubpixelConvWithClip(torch.nn.Module):
    """Upsample with last row and column clip."""

    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.PixelShuffle(2)

    def forward(self, *args):
        x = self.upsample(args[0])[:, :, :-1, :-1]
        x = torch.cat((x, x, x, x), dim=1)
        return x


class PumpAndDumpColumn(torch.nn.Module):
    epsilon = 0.1
    upsample_type = 'nearest'

    def __init__(self, in_features, pyramid_features, n_layers, *, block_factory, lateral_factory):
        super().__init__()

        self.bottlenecks = torch.nn.ModuleList([
            block_factory(pyramid_features)
            for _ in range(n_layers)
        ])

        LOG.info('pyramid upsample: %s', self.upsample_type)
        if self.upsample_type == 'nearest':
            self.upsample = UpsampleWithClip(scale_factor=2, mode='nearest')
        elif self.upsample_type == 'subpixel':
            self.upsample = SubpixelConvWithClip()
        else:
            raise Exception('upsample type unknown: {}'.format(self.upsample_type))

        self.lateral1 = torch.nn.ModuleList([
            lateral_factory(in_features, pyramid_features)
            for _ in range(n_layers + 1)
        ])
        self.lateral2_0 = lateral_factory(pyramid_features, pyramid_features)
        self.lateral2 = torch.nn.ModuleList([
            lateral_factory(pyramid_features, pyramid_features)
            for _ in range(n_layers)
        ])

        self.w_inputs1 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_inputs2_0 = torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
        self.w_inputs2 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_skips_0 = torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
        self.w_skips = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_dumpeds = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_pumpeds = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, pyramid_features, 1, 1)))
            for _ in range(n_layers)
        ])

    def forward(self, *args):
        inputs = args[0]
        # print([i.shape for i in inputs])

        # enforce positive weights
        for w in itertools.chain(self.w_inputs1, self.w_inputs2, self.w_skips,
                                 self.w_dumpeds, self.w_pumpeds):
            w.data.clamp_(0, 1e3)

        intermediate0 = [
            l(i)
            for i, l in czip(inputs, self.lateral1)
        ]

        intermediate1 = [intermediate0[0]]
        for input0, bottleneck, w_input1, w_pumped in czip(
                intermediate0[1:], self.bottlenecks,
                self.w_inputs1, self.w_pumpeds):
            pumped = bottleneck(intermediate1[-1])

            intermediate1.append(
                (w_input1 * input0 + w_pumped * pumped) / (
                    self.epsilon + w_input1 + w_pumped)
            )

        intermediate2 = [
            (
                self.w_inputs2_0 * self.lateral2_0(intermediate1[-1]) +
                self.w_skips_0 * intermediate0[-1]
            ) / (
                self.epsilon + self.w_inputs2_0 + self.w_skips_0
            )
        ]
        for input1, input2, lateral2, w_input2, w_dumped, w_skip in reversed_czip(
                intermediate0[:-1], intermediate1[:-1], self.lateral2,
                self.w_inputs2, self.w_dumpeds, self.w_skips):

            if hasattr(self, 'dequad'):  # backwards compat TODO remove
                dumped = self.dequad(intermediate2[0])[:, :, :-1, :-1]
                dumped = torch.cat((dumped, dumped, dumped, dumped), dim=1)
            else:
                dumped = self.upsample(intermediate2[0])

            input2_lateral = lateral2(input2)

            intermediate2.insert(
                0,
                (w_input2 * input2_lateral + w_dumped * dumped + w_skip * input1) / (
                    self.epsilon + w_input2 + w_dumped + w_skip)
            )

        return intermediate2
