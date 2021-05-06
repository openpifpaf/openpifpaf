import logging

import numpy as np

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class AddCrowdForIncompleteHead(Preprocess):
    @staticmethod
    def __call__(image, anns, meta):
        # detect incomplete annotations and add crowd annotations for those
        new_anns = []
        for ann in anns:
            if ann.get('iscrowd', False):
                continue
            if all(c > 0 for c in ann['keypoints'][0:3, 2]):
                continue

            if 'bbox_head' not in ann:
                LOG.warning('need to add crowd annotation but bbox_head not present')
                continue
            bbox = ann['bbox_head']
            new_anns.append({
                'image_id': ann['image_id'],
                'bbox': bbox.copy(),
                'keypoints': np.array([
                    (bbox[0], bbox[1], 0.0),
                    (bbox[0], bbox[1] + bbox[3], 0.0),
                    (bbox[2], bbox[1] + bbox[3], 0.0),
                    (bbox[2], bbox[1], 0.0),
                ], dtype=np.float32),
                'iscrowd': True,
                'track_id': -1,
            })

        return image, anns + new_anns, meta
