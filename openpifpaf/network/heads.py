"""Head networks."""

from abc import ABCMeta
import logging
import re

import torch

LOG = logging.getLogger(__name__)

HEADS = None


class Head(metaclass=ABCMeta):
    @classmethod
    def cli(cls, parser):
        """Add decoder specific command line arguments to the parser."""

    @classmethod
    def apply_args(cls, args):
        """Read command line arguments args to set class properties."""


class HeadStacks(torch.nn.Module):
    def __init__(self, stacks):
        super(HeadStacks, self).__init__()
        self.stacks_by_pos = {s[0]: s for s in stacks}
        self.ignore = {head_i for s in stacks for head_i in s[1:]}

    def forward(self, *args):
        heads = args

        stacked = []
        for head_i, head in enumerate(heads):
            if head_i in self.ignore:
                continue
            if head_i not in self.stacks_by_pos:
                stacked.append(head)
                continue

            fields = [heads[si] for si in self.stacks_by_pos[head_i]]
            stacked.append([
                torch.cat(fields_by_type, dim=1)
                for fields_by_type in zip(*fields)
            ])

        return stacked


class PifHFlip(torch.nn.Module):
    def __init__(self, keypoints, hflip):
        super(PifHFlip, self).__init__()

        flip_indices = torch.LongTensor([
            keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i
            for kp_i, kp_name in enumerate(keypoints)
        ])
        LOG.debug('hflip indices: %s', flip_indices)
        self.register_buffer('flip_indices', flip_indices)


    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)

        # flip the x-coordinate of the vector component
        out[1][:, :, 0, :, :] *= -1.0

        return out


class PafHFlip(torch.nn.Module):
    def __init__(self, keypoints, skeleton, hflip):
        super(PafHFlip, self).__init__()
        skeleton_names = [
            (keypoints[j1 - 1], keypoints[j2 - 1])
            for j1, j2 in skeleton
        ]
        flipped_skeleton_names = [
            (hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2)
            for j1, j2 in skeleton_names
        ]
        LOG.debug('skeleton = %s, flipped_skeleton = %s',
                  skeleton_names, flipped_skeleton_names)

        flip_indices = list(range(len(skeleton)))
        reverse_direction = []
        for paf_i, (n1, n2) in enumerate(skeleton_names):
            if (n1, n2) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n1, n2))
            if (n2, n1) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n2, n1))
                reverse_direction.append(paf_i)
        LOG.debug('hflip indices: %s, reverse: %s', flip_indices, reverse_direction)

        self.register_buffer('flip_indices', torch.LongTensor(flip_indices))
        self.register_buffer('reverse_direction', torch.LongTensor(reverse_direction))

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)

        # flip the x-coordinate of the vector components
        out[1][:, :, 0, :, :] *= -1.0
        out[2][:, :, 0, :, :] *= -1.0

        # reverse direction
        for paf_i in self.reverse_direction:
            cc = torch.clone(out[1][:, paf_i])
            out[1][:, paf_i] = out[2][:, paf_i]
            out[2][:, paf_i] = cc

        return out


class CompositeField(Head, torch.nn.Module):
    default_dropout_p = 0.0
    default_quad = 0
    default_kernel_size = 1
    default_padding = 0
    default_dilation = 1

    def __init__(self, head_name, in_features, *,
                 n_fields=None,
                 n_confidences=1, n_vectors=None, n_scales=None,
                 kernel_size=None, padding=None, dilation=None):
        super(CompositeField, self).__init__()

        n_fields = n_fields or self.determine_nfields(head_name)
        n_vectors = n_vectors or self.determine_nvectors(head_name)
        n_scales = n_scales or self.determine_nscales(head_name)
        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d',
                  head_name, n_fields, n_confidences, n_vectors, n_scales)

        if kernel_size is None:
            kernel_size = {'wpaf': 3}.get(head_name, self.default_kernel_size)
        if padding is None:
            padding = {'wpaf': 5}.get(head_name, self.default_padding)
        if dilation is None:
            dilation = {'wpaf': 5}.get(head_name, self.default_dilation)
        LOG.debug('%s config: kernel = %d, padding = %d, dilation = %d',
                  head_name, kernel_size, padding, dilation)

        self.shortname = head_name
        self.dilation = dilation

        self.dropout = torch.nn.Dropout2d(p=self.default_dropout_p)
        self._quad = self.default_quad

        # classification
        out_features = n_fields * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_confidences)
        ])

        # regression
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_vectors)
        ])
        self.reg_spreads = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in self.reg_convs
        ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    @staticmethod
    def determine_nfields(head_name):
        m = re.match('p[ia]f([0-9]+)$', head_name)
        if m is not None:
            return int(m.group(1))

        return {
            'paf': 19,
            'pafb': 19,
            'pafsb': 19,
            'pafs19': 19,
            'wpaf': 19,
        }.get(head_name, 17)

    @staticmethod
    def determine_nvectors(head_name):
        if 'pif' in head_name:
            return 1
        if 'paf' in head_name:
            return 2
        return 0

    @staticmethod
    def determine_nscales(head_name):
        if 'pif' in head_name:
            return 1
        if 'paf' in head_name:
            return 0
        return 0

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('head')
        group.add_argument('--head-dropout', default=cls.default_dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        group.add_argument('--head-quad', default=cls.default_quad, type=int,
                           help='number of times to apply quad (subpixel conv) to heads')
        group.add_argument('--head-kernel-size', default=cls.default_kernel_size, type=int,
                           help='[experimental]')
        group.add_argument('--head-padding', default=cls.default_padding, type=int,
                           help='[experimental]')
        group.add_argument('--head-dilation', default=cls.default_dilation, type=int,
                           help='[never-worked]')

    @classmethod
    def apply_args(cls, args):
        cls.default_dropout_p = args.head_dropout
        cls.default_quad = args.head_quad
        cls.default_kernel_size = args.head_kernel_size
        cls.default_padding = args.head_padding
        cls.default_dilation = args.head_dilation

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if not self.training:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self.reg_spreads]
        regs_x_spread = [torch.nn.functional.leaky_relu(x + 2.0) - 2.0
                         for x in regs_x_spread]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        # upscale
        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]

        # reshape regressions
        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x
