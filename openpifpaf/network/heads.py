"""Head networks."""

from dataclasses import dataclass
import functools
import logging
from typing import Any, List, Tuple, Union

import numpy as np
import torch

from ..panoptic_deeplab.decoder import panoptic_deeplab

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
            cif_head[:, :, 1:3].add_(index_field)
        if caf_head is not None:
            caf_head[:, :, 1:3].add_(index_field)
            caf_head[:, :, 3:5].add_(index_field)
            # rearrange caf_fields
            caf_head = caf_head[:, :, (0, 1, 2, 5, 7, 3, 4, 6, 8)]

        return cif_head, caf_head


class CifPanCollector(torch.nn.Module):
    def __init__(self, cif_indices):
        super(CifPanCollector, self).__init__()
        self.cif_indices = cif_indices
        LOG.debug('cif = %s', cif_indices)

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
        cif_heads = [self.concat_fields(self.selector(heads, head_index))
                     for head_index in self.cif_indices]

        # concat heads
        cif_head = self.concat_heads(cif_heads)

        # add index
        index_field = index_field_torch(cif_head.shape[-2:], device=cif_head.device)
        if cif_head is not None:
            cif_head[:, :, 1:3].add_(index_field)

        return cif_head, heads[1]


class CifdetCollector(torch.nn.Module):
    def __init__(self, indices):
        super(CifdetCollector, self).__init__()
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
    n_scales: int = 2

    @property
    def n_fields(self):
        return len(self.skeleton)


@dataclass
class DetectionMeta:
    name: str
    categories: List[str]

    n_confidences: int = 1
    n_vectors: int = 2
    n_scales: int = 0

    @property
    def n_fields(self):
        return len(self.categories)


@dataclass
class PanopticDeeplabMeta:
    """Inspired by the following Panoptic-Deeplab configuration (resnet50)

    DECODER:
        ~IN_CHANNELS: 2048~
        FEATURE_KEY: "res5"
        DECODER_CHANNELS: 256
        ATROUS_RATES: (3, 6, 9)
    PANOPTIC_DEEPLAB:
        LOW_LEVEL_CHANNELS: (1024, 512, 256)
        LOW_LEVEL_KEY: ["res4", "res3", "res2"]
        LOW_LEVEL_CHANNELS_PROJECT: (128, 64, 32)
        INSTANCE:
            ENABLE: True
            LOW_LEVEL_CHANNELS_PROJECT: (64, 32, 16)
            DECODER_CHANNELS: 128
            HEAD_CHANNELS: 32
            ASPP_CHANNELS: 256
            NUM_CLASSES: (1, 2)
            CLASS_KEY: ["center", "offset"]
    """

    name: str

    keypoints: List[str]
    skeleton: List[Tuple[int, int]]
    sparse_skeleton: List[Tuple[int, int]] = None

    feature_key: str = 'res5'
    decoder_channels: int = 256
    atrous_rates: Tuple[int] = (3,6,9)

    low_level_channels: Tuple[int] = (1024, 512, 256)
    low_level_key: Tuple[str] = ('res4', 'res3', 'res2')
    low_level_channels_project: Tuple[int] = (128, 64, 32)

    num_classes: Tuple[int] = (2, 2)
    class_key: Tuple[str] = ('semantic', 'offset')


### AMA
@dataclass
class SegmentationMeta:
    name: str
    keypoints: List[str]
    objects: List[str]
    # draw_skeleton: List[Tuple[int, int]] = None
    skeleton: List[Tuple[int, int]]
    sparse_skeleton: List[Tuple[int, int]] = None

    n_confidences: int = 1
    n_vectors: int = 1
    n_sigmas: int = 2

    @property
    def n_fields(self):
        return len(self.objects)


class PanopticDeeplabHead(torch.nn.Module):
    # TODO: Check configuration (dilation rates for os 16?)
    # TODO: What is the stride function of other heads?

    def __init__(self, meta:PanopticDeeplabMeta, in_channels):
        super().__init__()
        self.meta = meta

        self.decoder = panoptic_deeplab.SinglePanopticDeepLabDecoder(
            in_channels, meta.feature_key, meta.low_level_channels, meta.low_level_key,
            meta.low_level_channels_project, meta.decoder_channels, meta.atrous_rates,
            aspp_channels=None)
        self.head = panoptic_deeplab.SinglePanopticDeepLabHead(
            meta.decoder_channels, meta.decoder_channels, meta.num_classes, meta.class_key)

    def _upsample_predictions(self, pred, input_shape):
        # Override upsample method to correctly handle `offset`
        result = {}
        for key in pred.keys():
            out = torch.nn.functional.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def forward(self, x):
        # Use intermediate outputs as well
        features = x.all_outputs

        x = self.decoder(features)
        x = self.head(x)
        x = self._upsample_predictions(x, features['input'].shape[-2:])

        return x


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
        raise Exception('use CompositeFieldFused instead of CompositeField')

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
        # print('pif head')
        # print(regs_x.shape[2])
        # print(regs_x.shape[3])

        return classes_x + regs_x + regs_logb + scales_x


class CompositeFieldFused(torch.nn.Module):
    dropout_p = 0.0
    quad = 1

    def __init__(self,
                 meta: Union[IntensityMeta, AssociationMeta, DetectionMeta],
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
        feature_groups = [
            meta.n_confidences * meta.n_fields,
            meta.n_vectors * 2 * meta.n_fields,
            meta.n_vectors * 1 * meta.n_fields,
            meta.n_scales * meta.n_fields,
        ]
        self.out_features = []  # the cumulative of the feature_groups above

        for fg in feature_groups:
            self.out_features.append(
                (self.out_features[-1] if self.out_features else 0) + fg)
        self.conv = torch.nn.Conv2d(in_features, self.out_features[-1] * (4 ** self._quad),
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
            x = self.dequad_op(x)[:, :, :-1, :-1]

        # classification
        classes_x = x[:, 0:self.out_features[0]]
        classes_x = classes_x.view(classes_x.shape[0],
                                   self.meta.n_fields,
                                   self.meta.n_confidences,
                                   classes_x.shape[2],
                                   classes_x.shape[3])
        if not self.training:
            classes_x = torch.sigmoid(classes_x)

        # regressions
        regs_x = x[:, self.out_features[0]:self.out_features[1]]
        regs_x = regs_x.view(regs_x.shape[0],
                             self.meta.n_fields,
                             self.meta.n_vectors,
                             2,
                             regs_x.shape[2],
                             regs_x.shape[3])
        regs_logb = x[:, self.out_features[1]:self.out_features[2]]
        regs_logb = regs_logb.view(regs_logb.shape[0],
                                   self.meta.n_fields,
                                   self.meta.n_vectors,
                                   regs_logb.shape[2],
                                   regs_logb.shape[3])

        # scale
        scales_x = x[:, self.out_features[2]:self.out_features[3]]
        scales_x = scales_x.view(scales_x.shape[0],
                                 self.meta.n_fields,
                                 self.meta.n_scales,
                                 scales_x.shape[2],
                                 scales_x.shape[3])
        if not self.training:
            scales_x = torch.exp(scales_x)

        # print('pif head')
        # print(regs_x.shape[-1])
        # print(regs_x.shape[-2])

        return classes_x, regs_x, regs_logb, scales_x


### AMA
class InstanceSegHead(torch.nn.Module):
    dropout_p = 0.0
    quad = 1

    def __init__(self,
                 meta: SegmentationMeta,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__()

        # LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
        #           'kernel = %d, padding = %d, dilation = %d',
        #           meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
        #           kernel_size, padding, dilation)

        self.meta = meta
        self.dropout = torch.nn.Dropout2d(p=self.dropout_p)
        self._quad = self.quad

        feature_groups = [
            meta.n_confidences * meta.n_fields,
            meta.n_vectors * 2 * meta.n_fields,
            meta.n_sigmas * meta.n_fields,
        ]

        self.out_features = []  # the cumulative of the feature_groups above
        for fg in feature_groups:
            self.out_features.append(
                (self.out_features[-1] if self.out_features else 0) + fg)
        self.conv = torch.nn.Conv2d(in_features, self.out_features[-1] * (4 ** self._quad),
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
            x = self.dequad_op(x)[:, :, :-1, :-1]

        # seed
        seed_x = x[:, 0:self.out_features[0]]
        seed_x = seed_x.view(seed_x.shape[0],
                                   self.meta.n_fields,
                                   self.meta.n_confidences,
                                   seed_x.shape[2],
                                   seed_x.shape[3])
        if not self.training:
            seed_x = torch.sigmoid(seed_x)

        # embedding
        regs_x = x[:, self.out_features[0]:self.out_features[1]]
        regs_x = regs_x.view(regs_x.shape[0],
                             self.meta.n_fields,
                             self.meta.n_vectors,
                             2,
                             regs_x.shape[2],
                             regs_x.shape[3])


        # sigma
        sigma_x = x[:, self.out_features[1]:self.out_features[2]]
        sigma_x = sigma_x.view(sigma_x.shape[0],
                                self.meta.n_fields,
                                self.meta.n_sigmas,
                                sigma_x.shape[2],
                                sigma_x.shape[3])
        if not self.training:
            sigma_x = torch.tanh(sigma_x)

        # print('seg head factory')
        # print(seed_x.shape)
        # print(regs_x.shape)
        # print(sigma_x.shape)

        return seed_x, regs_x, sigma_x

