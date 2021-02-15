from abc import ABCMeta, abstractmethod
import copy
import math
import numpy as np

from ..annotation import AnnotationDet
from . import utils


class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""

    @staticmethod
    def keypoint_sets_inverse(keypoint_sets, meta):
        keypoint_sets = keypoint_sets.copy()

        keypoint_sets[:, :, 0] += meta['offset'][0]
        keypoint_sets[:, :, 1] += meta['offset'][1]

        keypoint_sets[:, :, 0] = keypoint_sets[:, :, 0] / meta['scale'][0]
        keypoint_sets[:, :, 1] = keypoint_sets[:, :, 1] / meta['scale'][1]

        if meta['hflip']:
            w = meta['width_height'][0]
            keypoint_sets[:, :, 0] = -keypoint_sets[:, :, 0] + (w - 1)
            for keypoints in keypoint_sets:
                if meta.get('horizontal_swap'):
                    keypoints[:] = meta['horizontal_swap'](keypoints)

        return keypoint_sets

    @staticmethod
    def annotations_inverse(annotations, meta):
        annotations = copy.deepcopy(annotations)

        # determine rotation parameters
        angle = -meta['rotation']['angle']
        rw = meta['rotation']['width']
        rh = meta['rotation']['height']
        cangle = math.cos(angle / 180.0 * math.pi)
        sangle = math.sin(angle / 180.0 * math.pi)

        for ann in annotations:
            if isinstance(ann, AnnotationDet):
                Preprocess.anndet_inverse(ann, meta)
                continue

            # rotation
            if angle != 0.0:
                xy = ann.data[:, :2]
                x_old = xy[:, 0].copy() - (rw - 1)/2
                y_old = xy[:, 1].copy() - (rh - 1)/2
                xy[:, 0] = (rw - 1)/2 + cangle * x_old + sangle * y_old
                xy[:, 1] = (rh - 1)/2 - sangle * x_old + cangle * y_old

            # offset
            ann.data[:, 0] += meta['offset'][0]
            ann.data[:, 1] += meta['offset'][1]

            # scale
            ann.data[:, 0] = ann.data[:, 0] / meta['scale'][0]
            ann.data[:, 1] = ann.data[:, 1] / meta['scale'][1]
            ann.joint_scales /= meta['scale'][0]

            assert not np.any(np.isnan(ann.data))

            if meta['hflip']:
                w = meta['width_height'][0]
                ann.data[:, 0] = -ann.data[:, 0] + (w - 1)
                if meta.get('horizontal_swap'):
                    ann.data[:] = meta['horizontal_swap'](ann.data)

            for _, __, c1, c2 in ann.decoding_order:
                c1[:2] += meta['offset']
                c2[:2] += meta['offset']

                c1[:2] /= meta['scale']
                c2[:2] /= meta['scale']

        return annotations

    @staticmethod
    def anndet_inverse(ann, meta):
        angle = -meta['rotation']['angle']
        if angle != 0.0:
            rw = meta['rotation']['width']
            rh = meta['rotation']['height']
            ann.bbox = utils.rotate_box(ann.bbox, rw - 1, rh - 1, angle)

        ann.bbox[:2] += meta['offset']
        ann.bbox[:2] /= meta['scale']
        ann.bbox[2:] /= meta['scale']
