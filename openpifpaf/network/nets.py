import logging
import torch

LOG = logging.getLogger(__name__)

MODEL_MIGRATION = set()


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, *,
                 process_input=None, process_heads=None):
        super().__init__()

        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads

        self.set_head_nets(head_nets)

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride

        self.head_nets = head_nets

    def forward(self, *args):
        image_batch = args[0]

        if self.process_input is not None:
            image_batch = self.process_input(image_batch)

        x = self.base_net(image_batch)
        head_outputs = [hn(x) for hn in self.head_nets]

        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)

        return head_outputs


class CrossTalk(torch.nn.Module):
    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, *args):
        image_batch = args[0]
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


class Shell2Scale(torch.nn.Module):
    def __init__(self, base_net, head_nets, *, reduced_stride=3):
        super().__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.reduced_stride = reduced_stride

    @staticmethod
    def merge_heads(original_h, reduced_h,
                    logb_component_indices,
                    stride):
        mask = reduced_h[0] > original_h[0][:, :,
                                            :stride * reduced_h[0].shape[2]:stride,
                                            :stride * reduced_h[0].shape[3]:stride]
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
                           :stride * reduced_c.shape[2]:stride,
                           :stride * reduced_c.shape[3]:stride][mask] = \
                    reduced_c[mask]
            elif len(original_c.shape) == 5:
                original_c[:, :, :,
                           :stride * reduced_c.shape[3]:stride,
                           :stride * reduced_c.shape[4]:stride][mask_vector] = \
                    reduced_c[mask_vector]
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


# pylint: disable=protected-access
def model_migration(net_cpu):
    model_defaults(net_cpu)

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    for m in net_cpu.modules():
        if not hasattr(m, '_non_persistent_buffers_set'):
            m._non_persistent_buffers_set = set()

    if not hasattr(net_cpu, 'head_nets') and hasattr(net_cpu, '_head_nets'):
        net_cpu.head_nets = net_cpu._head_nets

    for hn_i, hn in enumerate(net_cpu.head_nets):
        if not hn.meta.base_stride:
            hn.meta.base_stride = net_cpu.base_net.stride
        if hn.meta.head_index is None:
            hn.meta.head_index = hn_i
        if hn.meta.name == 'cif' and 'score_weights' not in vars(hn.meta):
            hn.meta.score_weights = [3.0] * 3 + [1.0] * (hn.meta.n_fields - 3)

    for mm in MODEL_MIGRATION:
        mm(net_cpu)


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
            m.eps = 1e-5

            # smaller step size for running std and mean update
            m.momentum = 0.01  # tf default is 0.99
            # m.momentum = 0.1  # pytorch default
