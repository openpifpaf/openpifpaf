import copy
import logging

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class MinSize(Preprocess):
    def __init__(self, min_side=1.0):
        self.min_side = min_side

    def __call__(self, image, anns, meta):
        anns = copy.deepcopy(anns)
        for ann in anns:
            if ann['bbox'][2] > self.min_side \
               and ann['bbox'][3] > self.min_side:
                continue
            ann['iscrowd'] = True

        return image, anns, meta
