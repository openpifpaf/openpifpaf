"""Head networks."""

from dataclasses import dataclass
import functools
import logging
import re
from typing import Any, List, Tuple, Union

import numpy as np
import torch

from ..data import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    KINEMATIC_TREE_SKELETON,
)

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
        super(CifCafCollector, self).__init__()
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
        # LOG.debug('fields = %s', [f.shape for f in fields])
        return torch.cat(
            [
                field if len(field.shape) == 5 else torch.unsqueeze(field, 2)
                for field in fields
            ],
            dim=2,
        )

    @staticmethod
    def concat_heads(heads):
        if len(heads) == 0:
            return None
        if len(heads) == 1:
            return heads[0]

        # LOG.debug('heads = %s', [h.shape for h in heads])
        return torch.cat(heads, dim=1)

    def forward(self, *args):
        heads = args[0]

        # concat fields
        cif_heads = [self.concat_fields(self.selector(heads, head_index))
                     for head_index in self.cif_indices]
        caf_heads = [self.concat_fields(self.selector(heads, head_index))
                     for head_index in self.caf_indices]

        # concat heads
        cif_head = self.concat_heads(cif_heads)
        caf_head = self.concat_heads(caf_heads)

        # add index
        index_field = index_field_torch(cif_head.shape[-2:], device=cif_head.device)
        if cif_head is not None:
            cif_head[:, :, 1:3] += index_field
        if caf_head is not None:
            caf_head[:, :, 1:3] += index_field
            caf_head[:, :, 3:5] += index_field
            # rearrange caf_fields
            caf_head = caf_head[:, :, (0, 1, 2, 5, 7, 3, 4, 6, 8)]

        return cif_head, caf_head


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


@dataclass
class IntensityMeta:
    name: str
    keypoints: List[str]
    sigmas: List[float]
    pose: Any
    draw_skeleton: List[Tuple[int, int]] = None

    n_confidences: int = 1
    n_vectors: int = 1
    n_scales: int = 1

    @property
    def n_fields(self):
        return len(self.keypoints)


@dataclass
class AssociationMeta:
    name: str
    keypoints: List[str]
    sigmas: List[float]
    pose: Any
    skeleton: List[Tuple[int, int]]
    sparse_skeleton: List[Tuple[int, int]] = None
    only_in_field_of_view: bool = False

    n_confidences: int = 1
    n_vectors: int = 2
    n_scales: int = 1

    @property
    def n_fields(self):
        return len(self.keypoints)


@dataclass
class DetectionMeta:
    name: str
    categories: List[str]

    n_confidences: int = 1
    n_vectors: int = 1
    n_scales: int = 2

    @property
    def n_fields(self):
        return len(self.categories)


class CompositeField(torch.nn.Module):
    dropout_p = 0.0
    quad = 1

    def __init__(self,
                 meta: Union[IntensityMeta, AssociationMeta, DetectionMeta],
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super(CompositeField, self).__init__()

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        self.meta = meta
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self._quad = self.quad

        # classification
        out_features = meta.n_fields * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(meta.n_confidences)
        ])

        # regression
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features,
                            kernel_size, padding=padding, dilation=dilation)
            for _ in range(meta.n_vectors)
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
            for _ in range(meta.n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    def stride(self, basenet_stride):
        return basenet_stride // (2 ** self._quad)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if not self.training:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) for reg_conv in self.reg_convs]
        regs_logb = [reg_spread(x) for reg_spread in self.reg_spreads]
        if self.training:
            regs_logb = [3.0 * torch.tanh(reg_logb / 3.0) for reg_logb in regs_logb]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        if not self.training:
            scales_x = [torch.exp(scale_x) for scale_x in scales_x]

        # upscale
        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_logb = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                         for reg_x_spread in regs_logb]
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

        return classes_x + regs_x + regs_logb + scales_x


def determine_meta(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, COCO_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             COCO_KEYPOINTS,
                             COCO_PERSON_SIGMAS,
                             COCO_UPRIGHT_POSE,
                             COCO_PERSON_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               DENSER_COCO_PERSON_CONNECTIONS,
                               sparse_skeleton=COCO_PERSON_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               COCO_PERSON_SKELETON)
    raise NotImplementedError


def factory(name, n_features):
    if re.match('[cp][ia]f([0-9]*)$', name) is None \
       and re.match('cifdet([0-9]*)$', name) is None:
        raise Exception('unknown head to create a head network: {}'.format(name))

    meta = determine_meta(name)
    return CompositeField(meta, n_features)
