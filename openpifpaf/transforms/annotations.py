import copy
import logging

import numpy as np
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class NormalizeAnnotations(Preprocess):
    @staticmethod
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        for ann in anns:
            if 'keypoints' not in ann:
                ann['keypoints'] = []

            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            if 'bbox_original' not in ann:
                ann['bbox_original'] = np.copy(ann['bbox'])
            if 'segmentation' in ann:
                del ann['segmentation']

        return anns

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
