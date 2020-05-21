import dataclasses
import logging

import numpy as np
import torch

from .annrescaler import AnnRescaler
from ..visualizer import Cif as CifVisualizer
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Cif:
    rescaler: AnnRescaler
    sigmas: list
    v_threshold: int = 0

    side_length: int = 4
    padding: int = 10
    visualizer: CifVisualizer = None

    def __call__(self, image, anns, meta):
        return CifGenerator(self)(image, anns, meta)


class CifGenerator(object):
    def __init__(self, config: Cif):
        self.config = config

        self.intensities = None
        self.fields_reg = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.config.rescaler.keypoint_sets(anns)
        bg_mask = self.config.rescaler.bg_mask(anns, width_height_original)
        valid_area = self.config.rescaler.valid_area(meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = keypoint_sets.shape[1]
        self.init_fields(n_fields, bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)

        self.config.visualizer.processed_image(image)
        self.config.visualizer.targets(fields, keypoint_sets=keypoint_sets)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 6, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg[:, 2:] = np.inf
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

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

    @classmethod
    def max_r(cls, xyv, other_xyv):
        out = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        if not other_xyv:
            return out

        other_xyv = np.asarray(other_xyv)
        diffs = other_xyv[:, :2] - np.expand_dims(xyv[:2], 0)
        qs = cls.quadrant(diffs)
        for q in range(4):
            if not np.any(qs == q):
                continue
            out[q] = np.min(np.linalg.norm(diffs[qs == q], axis=1))

        return out

    def fill_keypoints(self, keypoints, other_keypoints):
        scale = self.config.rescaler.scale(keypoints)
        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.config.v_threshold:
                continue

            other_xyv = [other_kps[f] for other_kps in other_keypoints
                         if other_kps[f, 2] > self.config.v_threshold]
            max_r = self.max_r(xyv, other_xyv)

            joint_scale = scale if self.config.sigmas is None else scale * self.config.sigmas[f]
            joint_scale = np.min([joint_scale, np.min(max_r) * 0.25])

            self.fill_coordinate(f, xyv, joint_scale, max_r)

    def fill_coordinate(self, f, xyv, scale, max_r):
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0

        # update regression
        patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
        patch[:2, mask] = sink_reg[:, mask]
        patch[2:, mask] = np.expand_dims(max_r, 1) * 0.5

        # update scale
        assert np.isnan(scale) or scale > 0.0
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_scale = self.fields_scale[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale, valid_area, fill_value=np.nan)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg),
            torch.from_numpy(fields_scale),
        )
