import logging

import numpy as np
import scipy.ndimage
import torch

from .annrescaler import AnnRescaler
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


def scale_from_keypoints(keypoints):
    visible = keypoints[:, 2] > 0
    if not np.any(visible):
        return np.nan

    area = (
        (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
        (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
    )
    scale = np.sqrt(area)
    if scale < 1.0:
        scale = np.nan

    LOG.debug('instance scale = %.3f', scale)
    return scale


class Pif(object):
    side_length = 4

    def __init__(self, stride, *, n_keypoints, sigmas, v_threshold=0):
        self.stride = stride
        self.n_keypoints = n_keypoints
        self.sigmas = sigmas
        self.v_threshold = v_threshold

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.side_length)

        n_fields = keypoint_sets.shape[1]
        f = PifGenerator(self.side_length, self.v_threshold, self.sigmas)
        f.init_fields(n_fields, bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class PifGenerator(object):
    def __init__(self, side_length, v_threshold, sigmas, padding=10):
        self.side_length = side_length
        self.v_threshold = v_threshold
        self.sigmas = sigmas
        self.padding = padding

        self.intensities = None
        self.fields_reg = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(side_length)
        self.s_offset = (side_length - 1.0) / 2.0

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.padding
        field_h = bg_mask.shape[0] + 2 * self.padding
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.zeros((n_fields, 6, field_h, field_w), dtype=np.float32)
        self.fields_reg[:, 2:] = np.inf
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.s_offset) + 1,
                                                            border_value=1)

    def fill(self, keypoint_sets):
        for kps_i, keypoints in enumerate(keypoint_sets):
            self.fill_keypoints(
                keypoints,
                [kps for i, kps in enumerate(keypoint_sets) if i != kps_i],
            )

    @staticmethod
    def quadrant(xys):
        q = np.zeros((xys.shape[0],), dtype=np.int)
        q[xys[:, 0] < 0.0] += 1
        q[xys[:, 1] < 0.0] += 2
        return q

    def fill_keypoints(self, keypoints, other_keypoints):
        visible = keypoints[:, 2] > 0
        if not np.any(visible):
            return
        scale = scale_from_keypoints(keypoints)

        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.v_threshold:
                continue

            max_r = [np.inf, np.inf, np.inf, np.inf]
            other_xyv = [other_kps[f] for other_kps in other_keypoints
                         if other_kps[f, 2] > self.v_threshold]
            if other_xyv:
                other_xyv = np.asarray(other_xyv)
                diffs = other_xyv[:, :2] - np.expand_dims(xyv[:2], 0)
                qs = self.quadrant(diffs)
                for q in range(4):
                    if not np.any(qs == q):
                        continue
                    max_r[q] = np.min(np.linalg.norm(diffs[qs == q], axis=1)) / 2.0

            max_r = np.expand_dims(max_r, 1)
            self.fill_coordinate(f, xyv, scale, max_r)

    def fill_coordinate(self, f, xyv, scale, max_r):
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

        # update regression
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
        patch[:2, mask] = sink_reg[:, mask]
        patch[2:, mask] = max_r
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale * self.sigmas[f]

    def fields(self, valid_area):
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg = self.fields_reg[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        mask_valid_area(intensities, valid_area)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg),
            torch.from_numpy(fields_scale),
        )
