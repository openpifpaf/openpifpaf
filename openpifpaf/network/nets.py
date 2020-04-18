import logging
import torch

from . import heads
from .. import datasets

LOG = logging.getLogger(__name__)


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, *,
                 process_heads=None, cross_talk=0.0):
        super(Shell, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.process_heads = process_heads
        self.cross_talk = cross_talk

    def forward(self, *args):
        image_batch = args[0]

        if self.training and self.cross_talk:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk

        x = self.base_net(image_batch)
        head_outputs = [hn(x) for hn in self.head_nets]

        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)

        return head_outputs


class Shell2Scale(torch.nn.Module):
    def __init__(self, base_net, head_nets, *, reduced_stride=3):
        super(Shell2Scale, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.reduced_stride = reduced_stride

    @staticmethod
    def merge_heads(original_h, reduced_h,
                    logb_component_indices,
                    stride):
        mask = reduced_h[0] > original_h[0][:, :,
                                            :stride*reduced_h[0].shape[2]:stride,
                                            :stride*reduced_h[0].shape[3]:stride]
        mask_vector = torch.stack((mask, mask), dim=2)

        for ci, (original_c, reduced_c) in enumerate(zip(original_h, reduced_h)):
            if ci == 0:
                # confidence component
                reduced_c = reduced_c * 0.5
            elif ci in logb_component_indices:
                # log(b) components
                reduced_c = torch.log(torch.exp(reduced_c) * stride)
            else:
                # vectorial and scale components
                reduced_c = reduced_c * stride

            if len(original_c.shape) == 4:
                original_c[:, :,
                           :stride*reduced_c.shape[2]:stride,
                           :stride*reduced_c.shape[3]:stride][mask] = reduced_c[mask]
            elif len(original_c.shape) == 5:
                original_c[:, :, :,
                           :stride*reduced_c.shape[3]:stride,
                           :stride*reduced_c.shape[4]:stride][mask_vector] = reduced_c[mask_vector]
            else:
                raise Exception('cannot process component with shape {}'
                                ''.format(original_c.shape))

    def forward(self, *args):
        original_input = args[0]
        original_x = self.base_net(original_input)
        original_heads = [hn(original_x) for hn in self.head_nets]

        reduced_input = original_input[:, :, ::self.reduced_stride, ::self.reduced_stride]
        reduced_x = self.base_net(reduced_input)
        reduced_heads = [hn(reduced_x) for hn in self.head_nets]

        logb_component_indices = [(2,), (3, 4)]

        for original_h, reduced_h, lci in zip(original_heads,
                                              reduced_heads,
                                              logb_component_indices):
            self.merge_heads(original_h, reduced_h, lci, self.reduced_stride)

        return original_heads


class ShellMultiScale(torch.nn.Module):
    def __init__(self, base_net, head_nets, *,
                 process_heads=None, include_hflip=True):
        super(ShellMultiScale, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.pif_hflip = heads.PifHFlip(
            head_nets[0].meta.keypoints, datasets.constants.HFLIP)
        self.paf_hflip = heads.PafHFlip(
            head_nets[1].meta.keypoints, head_nets[1].meta.skeleton, datasets.constants.HFLIP)
        self.paf_hflip_dense = heads.PafHFlip(
            head_nets[2].meta.keypoints, head_nets[2].meta.skeleton, datasets.constants.HFLIP)
        self.process_heads = process_heads
        self.include_hflip = include_hflip

    def forward(self, *args):
        original_input = args[0]

        head_outputs = []
        for hflip in ([False, True] if self.include_hflip else [False]):
            for reduction in [1, 1.5, 2, 3, 5]:
                if reduction == 1.5:
                    x_red = torch.ByteTensor(
                        [i % 3 != 2 for i in range(original_input.shape[3])])
                    y_red = torch.ByteTensor(
                        [i % 3 != 2 for i in range(original_input.shape[2])])
                    reduced_input = original_input[:, :, y_red, :]
                    reduced_input = reduced_input[:, :, :, x_red]
                else:
                    reduced_input = original_input[:, :, ::reduction, ::reduction]

                if hflip:
                    reduced_input = torch.flip(reduced_input, dims=[3])

                reduced_x = self.base_net(reduced_input)
                head_outputs += [hn(reduced_x) for hn in self.head_nets]

        if self.include_hflip:
            for mscale_i in range(5, 10):
                head_i = mscale_i * 3
                head_outputs[head_i] = self.pif_hflip(*head_outputs[head_i])
                head_outputs[head_i + 1] = self.paf_hflip(*head_outputs[head_i + 1])
                head_outputs[head_i + 2] = self.paf_hflip_dense(*head_outputs[head_i + 2])

        if self.process_heads is not None:
            head_outputs = self.process_heads(*head_outputs)

        return head_outputs


# pylint: disable=too-many-branches
def model_migration(net_cpu):
    model_defaults(net_cpu)

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None


def model_defaults(net_cpu):
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            m.eps = 1e-4  # tf default is 0.001
            # m.eps = 1e-5  # pytorch default

            # less momentum for variance and expectation
            m.momentum = 0.01  # tf default is 0.99
            # m.momentum = 0.1  # pytorch default
