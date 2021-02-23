"""Head networks."""

import argparse
import functools
import logging
import math

import torch

from .. import headmeta

LOG = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def index_field_torch(shape, *, device=None, unsqueeze=(0, 0)):
    assert len(shape) == 2
    fliprow = torch.arange(shape[1]).repeat(shape[0], 1)
    flipcol = torch.arange(shape[0]).repeat(shape[1], 1).t()
    xy = torch.cat([fliprow.unsqueeze(0), flipcol.unsqueeze(0)])

    if device is not None:
        xy = xy.to(device, non_blocking=True)

    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)

    return xy


class PifHFlip(torch.nn.Module):
    def __init__(self, keypoints, hflip):
        super().__init__()

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
        super().__init__()
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


class HeadNetwork(torch.nn.Module):
    """Base class for head networks.

    :param meta: head meta instance to configure this head network
    :param in_features: number of input features which should be equal to the
        base network's output features
    """
    def __init__(self, meta: headmeta.Base, in_features: int):
        super().__init__()
        self.meta = meta
        self.in_features = in_features

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""


class CompositeField3(HeadNetwork):
    dropout_p = 0.0
    inplace_ops = True

    def __init__(self,
                 meta: headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)

        # convolution
        out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        self.conv = torch.nn.Conv2d(in_features, out_features * (meta.upsample_stride ** 2),
                                    kernel_size, padding=padding, dilation=dilation)

        # upsample
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CompositeField3')
        group.add_argument('--cf3-dropout', default=cls.dropout_p, type=float,
                           help='[experimental] zeroing probability of feature in head input')
        assert cls.inplace_ops
        group.add_argument('--cf3-no-inplace-ops', dest='cf3_inplace_ops',
                           default=True, action='store_false',
                           help='alternative graph without inplace ops')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.dropout_p = args.cf3_dropout
        cls.inplace_ops = args.cf3_inplace_ops

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        # Extract some shape parameters once.
        # Convert to int so that shape is constant in ONNX export.
        x_size = x.size()
        batch_size = int(x_size[0])
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields,
            self.meta.n_confidences + self.meta.n_vectors * 3 + self.meta.n_scales,
            feature_height,
            feature_width
        )

        if not self.training and self.inplace_ops:
            # classification
            classes_x = x[:, :, 0:self.meta.n_confidences]
            torch.sigmoid_(classes_x)

            # regressions x: add index
            if self.meta.n_vectors > 0:
                index_field = index_field_torch(x.shape[-2:], device=x.device)
                first_reg_feature = self.meta.n_confidences
                for i, do_offset in enumerate(self.meta.vector_offsets):
                    if not do_offset:
                        continue
                    reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    reg_x.add_(index_field)

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x[:] = torch.nn.functional.softplus(scales_x)
        elif not self.training and not self.inplace_ops:
            # TODO: CoreMLv4 does not like strided slices.
            # Strides are avoided when switching the first and second dim
            # temporarily.
            x = torch.transpose(x, 1, 2)

            # classification
            classes_x = x[:, 0:self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)

            # regressions x
            first_reg_feature = self.meta.n_confidences
            regs_x = [
                x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                for i in range(self.meta.n_vectors)
            ]
            # regressions x: add index
            index_field = index_field_torch(x.shape[-2:], device=x.device, unsqueeze=(1, 0))
            regs_x = [torch.add(reg_x, index_field) if do_offset else reg_x
                      for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

            # regressions logb
            first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
            regs_logb = x[:, first_reglogb_feature:first_reglogb_feature + self.meta.n_vectors]

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.nn.functional.softplus(scales_x)

            # concat
            x = torch.cat([classes_x, *regs_x, regs_logb, scales_x], dim=1)

            # TODO: CoreMLv4 problem (see above).
            x = torch.transpose(x, 1, 2)

        return x
