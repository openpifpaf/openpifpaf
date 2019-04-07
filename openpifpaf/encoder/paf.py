import logging
import numpy as np
import scipy
import torch

from ..data import COCO_PERSON_SKELETON, DENSER_COCO_PERSON_SKELETON, KINEMATIC_TREE_SKELETON
from .annrescaler import AnnRescaler
from .encoder import Encoder
from .utils import create_sink, mask_valid_area


class Paf(Encoder):
    default_min_size = 3
    default_fixed_size = False
    default_aspect_ratio = 0.0

    def __init__(self, head_name, stride, *, skeleton=None, n_keypoints=17, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments in %s: %s', head_name, kwargs)

        if skeleton is None:
            if head_name in ('paf', 'paf19', 'pafs', 'wpaf', 'pafb'):
                skeleton = COCO_PERSON_SKELETON
            elif head_name in ('paf16',):
                skeleton = KINEMATIC_TREE_SKELETON
            elif head_name in ('paf44',):
                skeleton = DENSER_COCO_PERSON_SKELETON
            else:
                raise Exception('unknown skeleton type of head')

        self.stride = stride
        self.n_keypoints = n_keypoints
        self.skeleton = skeleton

        self.min_size = self.default_min_size
        self.fixed_size = self.default_fixed_size
        self.aspect_ratio = self.default_aspect_ratio

        if self.fixed_size:
            assert self.aspect_ratio == 0.0

    @staticmethod
    def match(head_name):
        return head_name in (
            'paf',
            'paf19',
            'paf16',
            'paf44',
            'pafs',
            'wpaf',
            'pafb',
        )

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('paf encoder')
        group.add_argument('--paf-min-size', default=cls.default_min_size, type=int,
                           help='min side length of the PAF field')
        group.add_argument('--paf-fixed-size', default=cls.default_fixed_size, action='store_true',
                           help='fixed paf size')
        group.add_argument('--paf-aspect-ratio', default=cls.default_aspect_ratio, type=float,
                           help='paf width relative to its length')

    @classmethod
    def apply_args(cls, args):
        cls.default_min_size = args.paf_min_size
        cls.default_fixed_size = args.paf_fixed_size
        cls.default_aspect_ratio = args.paf_aspect_ratio

    def __call__(self, anns, width_height_original):
        rescaler = AnnRescaler(self.stride, self.n_keypoints)
        keypoint_sets, bg_mask, valid_area = rescaler(anns, width_height_original)
        self.log.debug('valid area: %s, paf min size = %d', valid_area, self.min_size)

        f = PafGenerator(self.min_size, self.skeleton,
                         fixed_size=self.fixed_size, aspect_ratio=self.aspect_ratio)
        f.init_fields(bg_mask)
        f.fill(keypoint_sets)
        return f.fields(valid_area)


class PafGenerator(object):
    def __init__(self, min_size, skeleton, *,
                 v_threshold=0, padding=10, fixed_size=False, aspect_ratio=0.0):
        self.min_size = min_size
        self.skeleton = skeleton
        self.v_threshold = v_threshold
        self.padding = padding
        self.fixed_size = fixed_size
        self.aspect_ratio = aspect_ratio

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_scale = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        n_fields = len(self.skeleton)
        field_w = bg_mask.shape[1] + 2 * self.padding
        field_h = bg_mask.shape[0] + 2 * self.padding
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_reg2 = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # set background
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.min_size / 2.0) + 1,
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

        for i, (joint1i, joint2i) in enumerate(self.skeleton):
            joint1 = keypoints[joint1i - 1]
            joint2 = keypoints[joint2i - 1]
            if joint1[2] <= self.v_threshold or joint2[2] <= self.v_threshold:
                continue

            self.fill_association(i, joint1, joint2, scale)

    def fill_association(self, i, joint1, joint2, scale):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.min_size, int(offset_d * self.aspect_ratio))
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

            # update background
            self.intensities[-1, fminy:fmaxy, fminx:fmaxx] = 0.0

            # update regressions
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset
            sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
                                np.linalg.norm(sink2, axis=0))
            mask = sink_l < self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx]
            self.fields_reg1[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink1[:, mask]
            self.fields_reg2[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink2[:, mask]
            self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update scale
            self.fields_scale[i, fminy:fmaxy, fminx:fmaxx][mask] = scale

    def fields(self, valid_area):
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg1 = self.fields_reg1[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg2 = self.fields_reg2[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        intensities = mask_valid_area(intensities, valid_area)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg1),
            torch.from_numpy(fields_reg2),
            torch.from_numpy(fields_scale),
        )
