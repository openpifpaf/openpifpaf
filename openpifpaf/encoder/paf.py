import logging
import numpy as np
import scipy
import torch

from .annrescaler import AnnRescaler
from .pif import scale_from_keypoints, PifGenerator
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


class Paf(object):
    min_size = 3
    fixed_size = False
    aspect_ratio = 0.0

    def __init__(self, stride, *, n_keypoints, skeleton, sigmas,
                 sparse_skeleton=None,
                 dense_to_sparse_radius=2.0,
                 only_in_field_of_view=False,
                 v_threshold=0):
        self.stride = stride
        self.n_keypoints = n_keypoints
        self.skeleton = skeleton
        self.sigmas = sigmas
        self.sparse_skeleton = sparse_skeleton
        self.dense_to_sparse_radius = dense_to_sparse_radius
        self.only_in_field_of_view = only_in_field_of_view
        self.v_threshold = v_threshold

        if self.fixed_size:
            assert self.aspect_ratio == 0.0

        LOG.debug('stride = %d, keypoints = %d, only_in_field_of_view = %s',
                  stride, n_keypoints, only_in_field_of_view)

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        keypoint_sets = rescaler.keypoint_sets(anns)
        bg_mask = rescaler.bg_mask(anns, width_height_original)
        valid_area = rescaler.valid_area(meta)
        LOG.debug('valid area: %s, paf min size = %d', valid_area, self.min_size)

        f = PafGenerator(self.min_size, self.skeleton,
                         v_threshold=self.v_threshold,
                         fixed_size=self.fixed_size,
                         aspect_ratio=self.aspect_ratio,
                         sigmas=self.sigmas,
                         sparse_skeleton=self.sparse_skeleton,
                         dense_to_sparse_radius=self.dense_to_sparse_radius,
                         only_in_field_of_view=self.only_in_field_of_view)
        f.init_fields(bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class PafGenerator(object):
    def __init__(self, min_size, skeleton, *,
                 v_threshold, fixed_size, aspect_ratio, sigmas,
                 sparse_skeleton,
                 dense_to_sparse_radius,
                 only_in_field_of_view,
                 padding=10):
        self.min_size = min_size
        self.skeleton_m1 = np.asarray(skeleton) - 1
        self.v_threshold = v_threshold
        self.padding = padding
        self.fixed_size = fixed_size
        self.aspect_ratio = aspect_ratio
        self.sigmas = sigmas
        self.sparse_skeleton_m1 = np.asarray(sparse_skeleton) - 1 if sparse_skeleton else None
        self.dense_to_sparse_radius = dense_to_sparse_radius
        self.only_in_field_of_view = only_in_field_of_view

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        n_fields = len(self.skeleton_m1)
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

    def shortest_sparse(self, joint_i, keypoints):
        shortest = np.inf
        for joint1i, joint2i in self.sparse_skeleton_m1:
            if joint_i not in (joint1i, joint2i):
                continue

            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.v_threshold or joint2[2] <= self.v_threshold:
                continue

            d = np.linalg.norm(joint1[:2] - joint2[:2])
            shortest = min(d, shortest)

        return shortest

    def fill_keypoints(self, keypoints, other_keypoints):
        scale = scale_from_keypoints(keypoints)
        for paf_i, (joint1i, joint2i) in enumerate(self.skeleton_m1):
            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.v_threshold or joint2[2] <= self.v_threshold:
                continue

            # check if there are shorter connections in the sparse skeleton
            if self.sparse_skeleton_m1 is not None:
                d = np.linalg.norm(joint1[:2] - joint2[:2])
                if self.shortest_sparse(joint1i, keypoints) * self.dense_to_sparse_radius < d \
                   and self.shortest_sparse(joint2i, keypoints) * self.dense_to_sparse_radius < d:
                    continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            if self.only_in_field_of_view:
                # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
                if joint1[0] < 0 or \
                   joint2[0] < 0 or \
                   joint1[0] > self.intensities.shape[2] - 1 - 2 * self.padding or \
                   joint2[0] > self.intensities.shape[2] - 1 - 2 * self.padding:
                    continue
                if joint1[1] < 0 or \
                   joint2[1] < 0 or \
                   joint1[1] > self.intensities.shape[1] - 1 - 2 * self.padding or \
                   joint2[1] > self.intensities.shape[1] - 1 - 2 * self.padding:
                    continue

            other_j1s = [other_kps[joint1i] for other_kps in other_keypoints
                         if other_kps[joint1i, 2] > self.v_threshold]
            other_j2s = [other_kps[joint2i] for other_kps in other_keypoints
                         if other_kps[joint2i, 2] > self.v_threshold]
            max_r1 = PifGenerator.max_r(joint1, other_j1s)
            max_r2 = PifGenerator.max_r(joint2, other_j2s)

            if self.sigmas is None:
                scale1, scale2 = scale, scale
            else:
                scale1 = scale * self.sigmas[joint1i]
                scale2 = scale * self.sigmas[joint2i]
            scale1 = np.min([scale1, np.min(max_r1) * 0.25])
            scale2 = np.min([scale2, np.min(max_r2) * 0.25])
            self.fill_association(paf_i, joint1, joint2, scale1, scale2, max_r1, max_r2)

    def fill_association(self, paf_i, joint1, joint2, scale1, scale2, max_r1, max_r2):
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
            self.intensities[paf_i, fminy:fmaxy, fminx:fmaxx] = 1.0

            # update regressions
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset
            sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
                                np.linalg.norm(sink2, axis=0))
            mask = sink_l < self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx]
            patch1 = self.fields_reg1[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch1[:2, mask] = sink1[:, mask]
            patch1[2:, mask] = np.expand_dims(max_r1, 1) * 0.5
            patch2 = self.fields_reg2[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch2[:2, mask] = sink2[:, mask]
            patch2[2:, mask] = np.expand_dims(max_r2, 1) * 0.5
            self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update scale
            self.fields_scale1[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale1
            self.fields_scale2[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale2

    def fields(self, valid_area):
        p = self.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities[:-1], valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area)
        mask_valid_area(fields_reg1[:, 1], valid_area)
        mask_valid_area(fields_reg2[:, 0], valid_area)
        mask_valid_area(fields_reg2[:, 1], valid_area)
        mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg1),
            torch.from_numpy(fields_reg2),
            torch.from_numpy(fields_scale1),
            torch.from_numpy(fields_scale2),
        )
