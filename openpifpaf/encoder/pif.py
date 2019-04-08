import logging
import re

import numpy as np
import scipy.ndimage
import torch

from .annrescaler import AnnRescaler
from .encoder import Encoder
from .utils import create_sink, mask_valid_area


class Pif(Encoder):
    default_side_length = 4

    def __init__(self, head_name, stride, *, n_keypoints=None, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments in %s: %s', head_name, kwargs)

        self.stride = stride
        if n_keypoints is None:
            m = re.match('pif([0-9]+)$', head_name)
            if m is not None:
                n_keypoints = int(m.group(1))
                self.log.debug('using %d keypoints for pif', n_keypoints)
            else:
                n_keypoints = 17
        self.n_keypoints = n_keypoints
        self.side_length = self.default_side_length

    @staticmethod
    def match(head_name):
        return head_name in (
            'pif',
            'ppif',
            'pifb',
            'pifs',
        ) or re.match('pif([0-9]+)$', head_name) is not None

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('pif encoder')
        group.add_argument('--pif-side-length', default=cls.default_side_length, type=int,
                           help='side length of the PIF field')

    @classmethod
    def apply_args(cls, args):
        cls.default_side_length = args.pif_side_length

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        self.log.debug('valid area: %s, pif side length = %d', valid_area, self.side_length)

        n_fields = keypoint_sets.shape[1]
        f = PifGenerator(self.side_length)
        f.init_fields(n_fields, bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class PifGenerator(object):
    def __init__(self, side_length, v_threshold=0, padding=10):
        self.side_length = side_length
        self.v_threshold = v_threshold
        self.padding = padding

        self.intensities = None
        self.fields_reg = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(side_length)
        self.s_offset = (side_length - 1.0) / 2.0

        self.log = logging.getLogger(self.__class__.__name__)

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.padding
        field_h = bg_mask.shape[0] + 2 * self.padding
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.s_offset) + 1,
                                                            border_value=1)

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        visible = keypoints[:, 2] > 0
        if not np.any(visible):
            return

        area = (
            (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
            (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        scale = np.sqrt(area)
        self.log.debug('instance scale = %.3f', scale)

        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.v_threshold:
                continue

            self.fill_coordinate(f, xyv, scale)

    def fill_coordinate(self, f, xyv, scale):
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.side_length, miny + self.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.padding)
        offset = offset.reshape(2, 1, 1)

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx] = 1.0

        # allow unknown margin in background
        self.intensities[-1, miny:maxy, minx:maxx] = 0.0

        # update regression
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    def fields(self, valid_area):
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg = self.fields_reg[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        intensities = mask_valid_area(intensities, valid_area)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg),
            torch.from_numpy(fields_scale),
        )
