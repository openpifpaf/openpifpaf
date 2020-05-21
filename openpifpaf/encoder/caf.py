import dataclasses
import logging
import numpy as np
import torch

from .annrescaler import AnnRescaler
from .cif import CifGenerator
from ..visualizer import Caf as CafVisualizer
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Caf:
    rescaler: AnnRescaler
    skeleton: list
    sigmas: list
    sparse_skeleton: list = None
    dense_to_sparse_radius: float = 2.0
    only_in_field_of_view: bool = False
    v_threshold: int = 0

    min_size: int = 3
    fixed_size: bool = False
    aspect_ratio: float = 0.0
    padding: int = 10
    visualizer: CafVisualizer = None

    def __call__(self, image, anns, meta):
        return CafGenerator(self)(image, anns, meta)


class CafGenerator:
    def __init__(self, config: Caf):
        self.config = config
        self.skeleton_m1 = np.asarray(config.skeleton) - 1
        self.sparse_skeleton_m1 = (
            np.asarray(config.sparse_skeleton) - 1
            if config.sparse_skeleton else None)

        if self.config.fixed_size:
            assert self.config.aspect_ratio == 0.0

        LOG.debug('only_in_field_of_view = %s, paf min size = %d',
                  config.only_in_field_of_view,
                  self.config.min_size)

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None
        self.fields_reg_l = None

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.config.rescaler.keypoint_sets(anns)
        bg_mask = self.config.rescaler.bg_mask(anns, width_height_original)
        valid_area = self.config.rescaler.valid_area(meta)
        LOG.debug('valid area: %s', valid_area)

        self.init_fields(bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)

        self.config.visualizer.processed_image(image)
        self.config.visualizer.targets(fields, keypoint_sets=keypoint_sets)

        return fields

    def init_fields(self, bg_mask):
        n_fields = len(self.skeleton_m1)
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.full((n_fields, 6, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg2 = np.full((n_fields, 6, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg1[:, 2:] = np.inf
        self.fields_reg2[:, 2:] = np.inf
        self.fields_scale1 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale2 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
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

    def shortest_sparse(self, joint_i, keypoints):
        shortest = np.inf
        for joint1i, joint2i in self.sparse_skeleton_m1:
            if joint_i not in (joint1i, joint2i):
                continue

            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            d = np.linalg.norm(joint1[:2] - joint2[:2])
            shortest = min(d, shortest)

        return shortest

    def fill_keypoints(self, keypoints, other_keypoints):
        scale = self.config.rescaler.scale(keypoints)
        for paf_i, (joint1i, joint2i) in enumerate(self.skeleton_m1):
            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            # check if there are shorter connections in the sparse skeleton
            if self.sparse_skeleton_m1 is not None:
                d = np.linalg.norm(joint1[:2] - joint2[:2]) / self.config.dense_to_sparse_radius
                if self.shortest_sparse(joint1i, keypoints) < d \
                   and self.shortest_sparse(joint2i, keypoints) < d:
                    continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            if self.config.only_in_field_of_view:
                # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
                if joint1[0] < 0 or \
                   joint2[0] < 0 or \
                   joint1[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding or \
                   joint2[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding:
                    continue
                if joint1[1] < 0 or \
                   joint2[1] < 0 or \
                   joint1[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding or \
                   joint2[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding:
                    continue

            other_j1s = [other_kps[joint1i] for other_kps in other_keypoints
                         if other_kps[joint1i, 2] > self.config.v_threshold]
            other_j2s = [other_kps[joint2i] for other_kps in other_keypoints
                         if other_kps[joint2i, 2] > self.config.v_threshold]
            max_r1 = CifGenerator.max_r(joint1, other_j1s)
            max_r2 = CifGenerator.max_r(joint2, other_j2s)

            if self.config.sigmas is None:
                scale1, scale2 = scale, scale
            else:
                scale1 = scale * self.config.sigmas[joint1i]
                scale2 = scale * self.config.sigmas[joint2i]
            scale1 = np.min([scale1, np.min(max_r1) * 0.25])
            scale2 = np.min([scale2, np.min(max_r2) * 0.25])
            self.fill_association(paf_i, joint1, joint2, scale1, scale2, max_r1, max_r2)

    def fill_association(self, paf_i, joint1, joint2, scale1, scale2, max_r1, max_r2):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))
        # s = max(s, min(int(scale1), int(scale2)))
        sink = create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0-fmargin, num=num)
        if self.config.fixed_size:
            frange = [0.5]
        for f in frange:
            fij = np.round(joint1[:2] + f * offset - s_offset) + self.config.padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
            if fminx < 0 or fmaxx > self.intensities.shape[2] or \
               fminy < 0 or fmaxy > self.intensities.shape[1]:
                continue
            fxy = fij - self.config.padding + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset

            # mask
            # perpendicular distance computation:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            # Coordinate systems for this computation is such that
            # joint1 is at (0, 0).
            sink_l = np.fabs(
                offset[1] * sink1[0]
                - offset[0] * sink1[1]
            ) / (offset_d + 0.01)
            mask = sink_l < self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx]
            self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update intensity
            self.intensities[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = 1.0

            # update regressions
            patch1 = self.fields_reg1[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch1[:2, mask] = sink1[:, mask]
            patch1[2:, mask] = np.expand_dims(max_r1, 1) * 0.5
            patch2 = self.fields_reg2[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch2[:2, mask] = sink2[:, mask]
            patch2[2:, mask] = np.expand_dims(max_r2, 1) * 0.5

            # update scale
            assert np.isnan(scale1) or scale1 > 0.0
            self.fields_scale1[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale1
            assert np.isnan(scale2) or scale2 > 0.0
            self.fields_scale2[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale2

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg1),
            torch.from_numpy(fields_reg2),
            torch.from_numpy(fields_scale1),
            torch.from_numpy(fields_scale2),
        )
