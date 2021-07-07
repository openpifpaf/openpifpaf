import logging
import torch

LOG = logging.getLogger(__name__)


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, *,
                 process_input=None, process_heads=None):
        super().__init__()

        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads

        self.set_head_nets(head_nets)

    @property
    def head_metas(self):
        if self.head_nets is None:
            return None
        return [hn.meta for hn in self.head_nets]

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride

        self.head_nets = head_nets

    def forward(self, image_batch, *, head_mask=None):
        if self.process_input is not None:
            image_batch = self.process_input(image_batch)

        x = self.base_net(image_batch)
        if head_mask is not None:
            head_outputs = tuple(hn(x) if m else None for hn, m in zip(self.head_nets, head_mask))
        else:
            head_outputs = tuple(hn(x) for hn in self.head_nets)

        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)

        return head_outputs


class CrossTalk(torch.nn.Module):
    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, image_batch):
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


def model_defaults(net_cpu):
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            # m.eps = 1e-3  # tf default is 0.001
            # m.eps = 1e-5  # pytorch default

            # This epsilon only appears inside a sqrt in the denominator,
            # i.e. the effective epsilon for division is much bigger than the
            # given eps.
            # See equation here:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            m.eps = max(m.eps, 1e-3)  # mobilenetv3 actually has 1e-3

            # smaller step size for running std and mean update
            m.momentum = 0.01  # tf default is 0.99
            # m.momentum = 0.1  # pytorch default

        elif isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            m.eps = 1e-4

        elif isinstance(m, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d)):
            m.eps = 1e-4
            m.momentum = 0.01
