import logging
import numpy as np
import scipy
import torch

from .annrescaler import AnnRescaler
from .pif import scale_from_keypoints
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


class Paf(object):
    min_size = 3
    fixed_size = False
    aspect_ratio = 0.0

    def __init__(self, stride, *, n_keypoints, skeleton, sigmas, v_threshold=0):
        self.stride = stride
        self.n_keypoints = n_keypoints
        self.skeleton = skeleton
        self.sigmas = sigmas
        self.v_threshold = v_threshold

        if self.fixed_size:
            assert self.aspect_ratio == 0.0

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        LOG.debug('valid area: %s, paf min size = %d', valid_area, self.min_size)

        f = PafGenerator(self.min_size, self.skeleton,
                         v_threshold=self.v_threshold,
                         fixed_size=self.fixed_size,
                         aspect_ratio=self.aspect_ratio,
                         sigmas=self.sigmas)
        f.init_fields(bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class PafGenerator(object):
    def __init__(self, min_size, skeleton, *,
                 v_threshold, fixed_size, aspect_ratio, sigmas, padding=10):
        self.min_size = min_size
        self.skeleton = skeleton
        self.v_threshold = v_threshold
        self.padding = padding
        self.fixed_size = fixed_size
        self.aspect_ratio = aspect_ratio
        self.sigmas = sigmas

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        n_fields = len(self.skeleton)
        field_w = bg_mask.shape[1] + 2 * self.padding
        field_h = bg_mask.shape[0] + 2 * self.padding
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.zeros((n_fields, 6, field_h, field_w), dtype=np.float32)
        self.fields_reg2 = np.zeros((n_fields, 6, field_h, field_w), dtype=np.float32)
        self.fields_reg1[:, 2:] = np.inf
        self.fields_reg2[:, 2:] = np.inf
        self.fields_scale1 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale2 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # set background
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.min_size / 2.0) + 1,
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

        for i, (joint1i, joint2i) in enumerate(self.skeleton):
            joint1 = keypoints[joint1i - 1]
            joint2 = keypoints[joint2i - 1]
            if joint1[2] <= self.v_threshold or joint2[2] <= self.v_threshold:
                continue

            other_j1s = [other_kps[joint1i - 1] for other_kps in other_keypoints
                         if other_kps[joint1i - 1, 2] > self.v_threshold]
            other_j2s = [other_kps[joint2i - 1] for other_kps in other_keypoints
                         if other_kps[joint2i - 1, 2] > self.v_threshold]
            max_r1 = [np.inf, np.inf, np.inf, np.inf]
            max_r2 = [np.inf, np.inf, np.inf, np.inf]
            if other_j1s:
                other_j1s = np.asarray(other_j1s)
                diffs1 = other_j1s[:, :2] - np.expand_dims(joint1[:2], 0)
                qs1 = self.quadrant(diffs1)
                for q1 in range(4):
                    if not np.any(qs1 == q1):
                        continue
                    max_r1[q1] = np.min(np.linalg.norm(diffs1[qs1 == q1], axis=1)) / 2.0
            if other_j2s:
                other_j2s = np.asarray(other_j2s)
                diffs2 = other_j2s[:, :2] - np.expand_dims(joint2[:2], 0)
                qs2 = self.quadrant(diffs2)
                for q2 in range(4):
                    if not np.any(qs2 == q2):
                        continue
                    max_r2[q2] = np.min(np.linalg.norm(diffs2[qs2 == q2], axis=1)) / 2.0

            max_r1 = np.expand_dims(max_r1, 1)
            max_r2 = np.expand_dims(max_r2, 1)
            if self.sigmas is None:
                scale1, scale2 = scale, scale
            else:
                scale1 = scale * self.sigmas[joint1i - 1]
                scale2 = scale * self.sigmas[joint2i - 1]
            self.fill_association(i, joint1, joint2, scale1, scale2, max_r1, max_r2)

    def fill_association(self, i, joint1, joint2, scale1, scale2, max_r1, max_r2):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.min_size, int(offset_d * self.aspect_ratio))
        # s = max(s, min(int(scale1), int(scale2)))
        sink = create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # pixel coordinates of top-left joint pixel
        joint1ij = np.round(joint1[:2] - s_offset)
        joint2ij = np.round(joint2[:2] - s_offset)
        offsetij = joint2ij - joint1ij

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0-fmargin, num=num)
        if self.fixed_size:
            frange = [0.5]
        for f in frange:
            fij = np.round(joint1ij + f * offsetij) + self.padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
            if fminx < 0 or fmaxx > self.intensities.shape[2] or \
               fminy < 0 or fmaxy > self.intensities.shape[1]:
                continue
            fxy = (fij - self.padding) + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)

            # update intensity
            self.intensities[i, fminy:fmaxy, fminx:fmaxx] = 1.0

            # update regressions
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset
            sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
                                np.linalg.norm(sink2, axis=0))
            mask = sink_l < self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx]
            patch1 = self.fields_reg1[i, :, fminy:fmaxy, fminx:fmaxx]
            patch1[:2, mask] = sink1[:, mask]
            patch1[2:, mask] = max_r1
            patch2 = self.fields_reg2[i, :, fminy:fmaxy, fminx:fmaxx]
            patch2[:2, mask] = sink2[:, mask]
            patch2[2:, mask] = max_r2
            self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update scale
            self.fields_scale1[i, fminy:fmaxy, fminx:fmaxx][mask] = scale1
            self.fields_scale2[i, fminy:fmaxy, fminx:fmaxx][mask] = scale2

    def fields(self, valid_area):
        p = self.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg1),
            torch.from_numpy(fields_reg2),
            torch.from_numpy(fields_scale1),
            torch.from_numpy(fields_scale2),
        )
