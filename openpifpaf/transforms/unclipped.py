import copy
import logging

import numpy as np

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class UnclippedSides(Preprocess):
    def __init__(self, *, margin=10, clipped_sides_okay=2):
        self.margin = margin
        self.clipped_sides_okay = clipped_sides_okay

    def __call__(self, image, anns, meta):
        anns = copy.deepcopy(anns)
        meta_rb = (meta['valid_area'][0] + meta['valid_area'][2],
                   meta['valid_area'][1] + meta['valid_area'][3])
        for ann in anns:
            ann_rb = (ann['bbox'][0] + ann['bbox'][2],
                      ann['bbox'][1] + ann['bbox'][3])
            clipped_sides = 0
            if ann['bbox'][0] - meta['valid_area'][0] < self.margin:
                clipped_sides += 1
            if ann['bbox'][1] - meta['valid_area'][1] < self.margin:
                clipped_sides += 1
            if meta_rb[0] - ann_rb[0] < self.margin:
                clipped_sides += 1
            if meta_rb[1] - ann_rb[1] < self.margin:
                clipped_sides += 1

            if clipped_sides <= self.clipped_sides_okay:
                continue
            ann['iscrowd'] = True

        return image, anns, meta


class UnclippedArea(Preprocess):
    def __init__(self, *, threshold=0.5):
        self.threshold = threshold

    def __call__(self, image, anns, meta):
        anns = copy.deepcopy(anns)
        for ann in anns:
            area_original = np.prod(ann['bbox_original'][2:])
            area_origscale = np.prod(ann['bbox'][2:] / meta['scale'])
            LOG.debug('clipped = %.0f, orig = %.0f', area_origscale, area_original)

            if area_original > 0.0 and area_origscale / area_original > self.threshold:
                continue

            ann['iscrowd'] = True

        return image, anns, meta
