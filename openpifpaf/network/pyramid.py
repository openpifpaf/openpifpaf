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


class SimplePyramid(torch.nn.Module):
    n_layers = 3
    concat_features = 512

    def __init__(self, in_features, *, block_factory, lateral_factory):
        super().__init__()

        self.blocks = torch.nn.ModuleList([
            block_factory(in_features)
            for _ in range(self.n_layers)
        ])

        self.lateral1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_features, self.concat_features,
                    kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(self.concat_features),
                torch.nn.ReLU(),
            )
            for _ in range(self.n_layers + 1)
        ])

        self.lateral2 = torch.nn.ModuleList([
            lateral_factory(self.concat_features)
            for _ in range(self.n_layers + 1)
        ])

        self.upsample = UpsampleWithClip(scale_factor=2, mode='nearest')
        self.out_lateral = lateral_factory(self.concat_features * (self.n_layers + 1))

    def forward(self, *args):
        x = [args[0]]
        for block in self.blocks:
            x.append(block(x[-1]))

        x = [
            lateral1(xx) for lateral1, xx in czip(self.lateral1, x)
        ]

        x = [
            lateral2(xx) for lateral2, xx in czip(self.lateral2, x)
        ]

        concatenated = x[-1]
        for xx in reversed(x[:-1]):
            upsampled = self.upsample(concatenated)
            concatenated = torch.cat((upsampled, xx), dim=1)

        return self.out_lateral(concatenated)


class PumpAndDump(torch.nn.Module):
    n_layers = 3
    stack_size = 1
    n_features = 512

    def __init__(self, in_features, *, block_factory, lateral_factory):
        super().__init__()
        LOG.info('layers=%d, stack=%d', self.n_layers, self.stack_size)

        self.columns = torch.nn.ModuleList([
            InputColumn(in_features, self.n_features, self.n_layers)
        ] + [
            PumpAndDumpColumn(self.n_features,
                              self.n_layers,
                              block_factory=block_factory,
                              lateral_factory=lateral_factory)
            for s in range(self.stack_size)
        ] + [
            ConcatenateColumn(self.n_features, self.n_layers,
                              lateral_factory=lateral_factory)
        ])

        self.out_lateral = lateral_factory(self.n_features * (self.n_layers + 1))

    def forward(self, *args):
        x = [args[0]] + [
            args[0][:, :, 0::2**(l + 1), 0::2**(l + 1)]
            for l, _ in enumerate(self.columns[1].bottlenecks)
        ]

        for column in self.columns:
            x = column(x)

        return self.out_lateral(x[0])

    @staticmethod
    def create_bottleneck(n_features):
        return torchvision.models.resnet.Bottleneck(
            n_features, n_features // 4,
            stride=2,
            downsample=torch.nn.Sequential(
                torchvision.models.resnet.conv1x1(n_features, n_features, stride=2),
                torch.nn.BatchNorm2d(n_features),
                torch.nn.ReLU(),
            ),
        )

    @staticmethod
    def create_lateral(n_features):
        return torchvision.models.resnet.Bottleneck(
            n_features, n_features // 4,
        )

    @staticmethod
    def create_invertedresidual(n_features):
        return torchvision.models.shufflenetv2.InvertedResidual(
            n_features, n_features, stride=2)

    @staticmethod
    def create_lateral_invertedresidual(n_features):
        return torchvision.models.shufflenetv2.InvertedResidual(
            n_features, n_features, stride=1)


class InputColumn(torch.nn.Module):
    def __init__(self, in_features, out_features, n_layers):
        super().__init__()

        self.lateral = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
            )
            for _ in range(n_layers + 1)
        ])

    def forward(self, *args):
        inputs = args[0]
        return [l(i) for l, i in czip(self.lateral, inputs)]


class ConcatenateColumn(torch.nn.Module):
    def __init__(self, n_features, n_layers, *, lateral_factory):
        super().__init__()

        self.upsample = UpsampleWithClip(scale_factor=2, mode='nearest')

        self.lateral = torch.nn.ModuleList([
            lateral_factory(n_features)
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
    weighted = False

    def __init__(self, n_features, n_layers, *, block_factory, lateral_factory):
        super().__init__()

        self.bottlenecks = torch.nn.ModuleList([
            block_factory(n_features)
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
            lateral_factory(n_features)
            for _ in range(n_layers + 1)
        ])
        self.lateral2_0 = lateral_factory(n_features)
        self.lateral2 = torch.nn.ModuleList([
            lateral_factory(n_features)
            for _ in range(n_layers)
        ])

        self.w_inputs1 = [1.0 for _ in range(n_layers)]
        self.w_inputs2_0 = 1.0
        self.w_inputs2 = [1.0 for _ in range(n_layers)]
        self.w_skips_0 = 1.0
        self.w_skips = [1.0 for _ in range(n_layers)]
        self.w_dumpeds = [1.0 for _ in range(n_layers)]
        self.w_pumpeds = [1.0 for _ in range(n_layers)]
        if self.weighted:
            self.w_inputs1 = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
                for _ in range(n_layers)
            ])
            self.w_inputs2_0 = torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
            self.w_inputs2 = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
                for _ in range(n_layers)
            ])
            self.w_skips_0 = torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
            self.w_skips = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
                for _ in range(n_layers)
            ])
            self.w_dumpeds = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
                for _ in range(n_layers)
            ])
            self.w_pumpeds = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones((1, n_features, 1, 1)))
                for _ in range(n_layers)
            ])

    def forward(self, *args):
        inputs = args[0]
        # print([i.shape for i in inputs])

        # enforce positive weights
        for w in itertools.chain(self.w_inputs1,
                                 [self.w_inputs2_0], self.w_inputs2,
                                 [self.w_skips_0], self.w_skips,
                                 self.w_dumpeds, self.w_pumpeds):
            if isinstance(w, float):
                continue
            w.data.clamp_(0, 1e3)

        # input stage
        intermediate0 = [
            l(i)
            for i, l in czip(inputs, self.lateral1)
        ]

        # pump path
        intermediate1 = [intermediate0[0]]
        for intermediate0i, bottleneck, w_input1, w_pumped in czip(
                intermediate0[1:], self.bottlenecks,
                self.w_inputs1, self.w_pumpeds):
            pumped = bottleneck(intermediate1[-1])

            intermediate1.append(
                (w_input1 * intermediate0i + w_pumped * pumped) / (
                    self.epsilon + w_input1 + w_pumped)
            )

        # dump path
        intermediate2 = [
            (
                self.w_inputs2_0 * self.lateral2_0(intermediate1[-1]) +
                self.w_skips_0 * inputs[-1]
            ) / (
                self.epsilon + self.w_inputs2_0 + self.w_skips_0
            )
        ]
        for inputs_i, intermediate1i, lateral2, w_input2, w_dumped, w_skip in reversed_czip(
                inputs[:-1], intermediate1[:-1], self.lateral2,
                self.w_inputs2, self.w_dumpeds, self.w_skips):
            dumped = self.upsample(intermediate2[0])

            intermediate2.insert(
                0,
                (
                    w_input2 * lateral2(intermediate1i) +
                    w_dumped * dumped +
                    w_skip * inputs_i
                ) / (self.epsilon + w_input2 + w_dumped + w_skip)
            )

        return intermediate2
