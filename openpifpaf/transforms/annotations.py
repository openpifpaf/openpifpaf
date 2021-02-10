import copy
import logging

import numpy as np
import torch

from .. import annotation
from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class NormalizeAnnotations(Preprocess):
    @classmethod
    def normalize_annotations(cls, anns):
        anns = copy.deepcopy(anns)

        for ann in anns:
            if isinstance(ann, annotation.Base):
                # already converted to an annotation type
                continue

            if 'keypoints' not in ann:
                ann['keypoints'] = []
            if 'iscrowd' not in ann:
                ann['iscrowd'] = False

            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            if 'bbox' not in ann:
                ann['bbox'] = cls.bbox_from_keypoints(ann['keypoints'])
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            if 'bbox_original' not in ann:
                ann['bbox_original'] = np.copy(ann['bbox'])
            if 'segmentation' in ann:
                del ann['segmentation']

        return anns

    @staticmethod
    def bbox_from_keypoints(keypoints):
        visible_keypoints = keypoints[keypoints[:, 2] > 0.0]
        if not visible_keypoints.shape[0]:
            return [0, 0, 0, 0]

        x1 = np.min(visible_keypoints[:, 0])
        y1 = np.min(visible_keypoints[:, 1])
        x2 = np.max(visible_keypoints[:, 0])
        y2 = np.max(visible_keypoints[:, 1])
        return [x1, y1, x2 - x2, y2 - y1]

    def __call__(self, image, anns, meta):
        anns = self.normalize_annotations(anns)

        if meta is None:
            meta = {}

        # fill meta with defaults if not already present
        w, h = image.size
        meta_from_image = {
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0, w - 1, h - 1)),
            'hflip': False,
            'width_height': np.array((w, h)),
        }
        for k, v in meta_from_image.items():
            if k not in meta:
                meta[k] = v

        return image, anns, meta


class AnnotationJitter(Preprocess):
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        for ann in anns:
            keypoints_xy = ann['keypoints'][:, :2]
            sym_rnd_kp = (torch.rand(*keypoints_xy.shape).numpy() - 0.5) * 2.0
            keypoints_xy += self.epsilon * sym_rnd_kp

            sym_rnd_bbox = (torch.rand((4,)).numpy() - 0.5) * 2.0
            ann['bbox'] += 0.5 * self.epsilon * sym_rnd_bbox

        return image, anns, meta
