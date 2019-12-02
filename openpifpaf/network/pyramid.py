import itertools
import logging
import torch
import torchvision

LOG = logging.getLogger(__name__)


class PumpAndDump(torch.nn.Module):
    n_layers = 2
    stack_size = 1

    def __init__(self, in_features, block_factory=None):
        super().__init__()
        LOG.info('layers=%d, stack=%d', self.n_layers, self.stack_size)

        self.columns = torch.nn.ModuleList([
            PumpAndDumpColumn(in_features, self.n_layers, block_factory=block_factory)
            for _ in range(self.stack_size)
        ])

    def forward(self, *args):
        x = [args[0]] + [
            args[0][:, :, 0::2**(l + 1), 0::2**(l + 1)]
            for l, _ in enumerate(self.columns[0].bottlenecks)
        ]

        for column in self.columns:
            x = column(x)

        return x[0]


class PumpAndDumpColumn(torch.nn.Module):
    epsilon = 0.1

    def __init__(self, in_features, n_layers, block_factory=None):
        super().__init__()
        block_factory = block_factory or self.create_bottleneck

        self.bottlenecks = torch.nn.ModuleList([
            block_factory(in_features)
            for _ in range(n_layers)
        ])

        self.dequad = torch.nn.PixelShuffle(2)

        self.w_inputs1 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, in_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_inputs2 = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, in_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_skips = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((1, in_features, 1, 1)), requires_grad=False)
        ] + [
            torch.nn.Parameter(torch.ones((1, in_features, 1, 1)))
            for _ in range(n_layers - 1)
        ])
        self.w_dumpeds = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((1, in_features, 1, 1)))
            for _ in range(n_layers)
        ])
        self.w_pumpeds = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, in_features, 1, 1)))
            for _ in range(n_layers)
        ])

    @staticmethod
    def create_bottleneck(in_features):
        return torchvision.models.resnet.Bottleneck(
            in_features, in_features // 4,
            stride=2,
            downsample=torch.nn.Sequential(
                torchvision.models.resnet.conv1x1(in_features, in_features, stride=2),
                torch.nn.BatchNorm2d(in_features),
            ),
        )

    @staticmethod
    def create_invertedresidual(in_features):
        return torchvision.models.shufflenetv2.InvertedResidual(
            in_features, in_features, stride=2)

    def forward(self, *args):
        inputs = args[0]
        # print([i.shape for i in inputs])

        # enforce positive weights
        for w in itertools.chain(self.w_inputs1, self.w_inputs2, self.w_skips,
                                 self.w_dumpeds, self.w_pumpeds):
            w.data.clamp_(0, 1e3)

        # pumpeds = [bottleneck(i) for bottleneck, i in zip(self.bottlenecks, inputs[:-1])]
        # assert len(self.w_inputs1) == len(inputs[1:]) == len(self.w_pumpeds)
        # print(self.w_inputs1[0].shape)
        # print(inputs[1].shape)
        # print(self.w_pumpeds[0].shape)
        # print(pumpeds[0].shape)
        # intermediate1 = [inputs[0]] + [
        #     w_input1 * input + w_pumped * pumped / (
        #         self.epsilon + w_input1 + w_pumped)
        #     for w_input1, input, w_pumped, pumped in zip(
        #         self.w_inputs1, inputs[1:],
        #         self.w_pumpeds, pumpeds)
        # ]
        intermediate1 = [inputs[0]]
        for input1, bottleneck, w_input1, w_pumped in zip(
                inputs[1:], self.bottlenecks,
                self.w_inputs1, self.w_pumpeds):
            pumped = bottleneck(intermediate1[-1])
            intermediate1.append(
                (w_input1 * input1 + w_pumped * pumped) / (
                    self.epsilon + w_input1 + w_pumped)
            )



        # dumpeds = [self.dequad(i)[:, :, :-1, :-1] for i in intermediate1[1:]]
        # dumpeds = [torch.cat((d, d, d, d), dim=1) for d in dumpeds]
        # assert len(self.w_inputs2) == len(intermediate1[:-1]) == len(self.w_dumpeds)
        # print(self.w_inputs2[0].shape)
        # print(intermediate1[0].shape)
        # print(self.w_dumpeds[0].shape)
        # print(dumpeds[0].shape)
        # intermediate2 = [
        #     w_skip * input + w_input2 * input2 + w_dumped * dumped / (
        #         self.epsilon + w_skip + w_input2 + w_dumped)
        #     for w_skip, input, w_input2, input2, w_dumped, dumped in zip(
        #         self.w_skips, inputs[:-1],
        #         self.w_inputs2, intermediate1[:-1],
        #         self.w_dumpeds, dumpeds)
        # ] + [intermediate1[-1]]
        intermediate2 = [intermediate1[-1]]
        for input1, input2, w_input2, w_dumped, w_skip in zip(
                reversed(inputs[:-1]), reversed(intermediate1[:-1]),
                reversed(self.w_inputs2), reversed(self.w_dumpeds), reversed(self.w_skips)):
            dumped = self.dequad(intermediate2[0])[:, :, :-1, :-1]
            dumped = torch.cat((dumped, dumped, dumped, dumped), dim=1)
            intermediate2.insert(
                0,
                (w_input2 * input2 + w_dumped * dumped + w_skip * input1) / (
                    self.epsilon + w_input2 + w_dumped + w_skip)
            )

        return intermediate2
