"""Head networks."""

import functools
import logging

import numpy as np
import torch

from . import headmeta

LOG = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def index_field_torch(shape, *, device=None, n_unsqueeze=2):
    yx = np.indices(shape, dtype=np.float32)
    xy = np.flip(yx, axis=0)

    xy = torch.from_numpy(xy.copy())
    if device is not None:
        xy = xy.to(device, non_blocking=True)

    for _ in range(n_unsqueeze):
        xy = torch.unsqueeze(xy, 0)

    return xy


class CifCafCollector(torch.nn.Module):
    def __init__(self, cif_indices, caf_indices):
        super().__init__()
        self.cif_indices = cif_indices
        self.caf_indices = caf_indices
        LOG.debug('cif = %s, caf = %s', cif_indices, caf_indices)

    @staticmethod
    def selector(inputs, index):
        if not isinstance(index, (list, tuple)):
            return inputs[index]

        for ind in index:
            inputs = inputs[ind]
        return inputs

    @staticmethod
    def concat_fields(fields):
        fields = [
            f.view(f.shape[0], f.shape[1], f.shape[2] * f.shape[3], *f.shape[4:])
            if len(f.shape) == 6
            else f.view(f.shape[0], f.shape[1], f.shape[2], *f.shape[3:])
            for f in fields
        ]
        return torch.cat(fields, dim=2)

    @staticmethod
    def concat_heads(heads):
        if not heads:
            return None
        if len(heads) == 1:
            return heads[0]

        # LOG.debug('heads = %s', [h.shape for h in heads])
        return torch.cat(heads, dim=1)

    def forward(self, *args):
        heads = args[0]

        # concat heads
        cif_head = self.concat_heads([self.selector(heads, head_index)
                                      for head_index in self.cif_indices])
        caf_head = self.concat_heads([self.selector(heads, head_index)
                                      for head_index in self.caf_indices])

        # add index
        index_field = index_field_torch(cif_head.shape[-2:], device=cif_head.device)
        if cif_head is not None:
            cif_head[:, :, 1:3].add_(index_field)
        if caf_head is not None:
            caf_head[:, :, 1:3].add_(index_field)
            caf_head[:, :, 3:5].add_(index_field)

        return cif_head, caf_head


class CifdetCollector(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices
        LOG.debug('cifdet = %s', indices)

    @staticmethod
    def selector(inputs, index):
        if not isinstance(index, (list, tuple)):
            return inputs[index]

        for ind in index:
            inputs = inputs[ind]
        return inputs

    @staticmethod
    def concat_fields(fields):
        fields = [
            f.view(f.shape[0], f.shape[1], f.shape[2] * f.shape[3], *f.shape[4:])
            if len(f.shape) == 6
            else f.view(f.shape[0], f.shape[1], f.shape[2], *f.shape[3:])
            for f in fields
        ]
        return torch.cat(fields, dim=2)

    @staticmethod
    def concat_heads(heads):
        if not heads:
            return None
        if len(heads) == 1:
            return heads[0]

        # LOG.debug('heads = %s', [h.shape for h in heads])
        return torch.cat(heads, dim=1)

    def forward(self, *args):
        heads = args[0]

        # concat fields
        cifdet_heads = [self.concat_fields(self.selector(heads, head_index))
                        for head_index in self.indices]

        # concat heads
        cifdet_head = self.concat_heads(cifdet_heads)

        # add index
        index_field = index_field_torch(cifdet_head.shape[-2:], device=cifdet_head.device)
        cifdet_head[:, :, 1:3] += index_field
        # rearrange caf_fields
        cifdet_head = cifdet_head[:, :, (0, 1, 2, 5, 3, 4, 6)]

        return (cifdet_head,)


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


class CompositeField3(torch.nn.Module):
    dropout_p = 0.0
    quad = 1
    inplace_ops = True

    def __init__(self,
                 meta: headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__()

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        self.meta = meta
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self._quad = self.quad

        # convolution
        out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        self.conv = torch.nn.Conv2d(in_features, out_features * (4 ** self._quad),
                                    kernel_size, padding=padding, dilation=dilation)

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    @property
    def sparse_task_parameters(self):
        return [self.conv.weight]

    def stride(self, basenet_stride):
        return basenet_stride // (2 ** self._quad)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        # upscale
        for _ in range(self._quad):
            x = self.dequad_op(x)
            if self.training:
                x = x[:, :, :-1, :-1]  # negative axes not supported by ONNX TensorRT
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, :int(x.shape[2]) - 1, :int(x.shape[3]) - 1]

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

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            torch.exp_(scales_x)
        elif not self.training and not self.inplace_ops:
            # classification
            classes_x = x[:, :, 0:self.meta.n_confidences]
            classes_x = torch.sigmoid(classes_x)

            # regressions
            first_reg_feature = self.meta.n_confidences
            regs_x = x[:, :, first_reg_feature:first_reg_feature + self.meta.n_vectors * 3]

            # scale
            first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
            scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]
            scales_x = torch.exp(scales_x)

            # concat
            x = torch.cat([classes_x, regs_x, scales_x], dim=2)

        return x
