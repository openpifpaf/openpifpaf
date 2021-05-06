import copy
import logging

import PIL
import torch

from ..preprocess import Preprocess

LOG = logging.getLogger(__name__)


class CameraShift(Preprocess):
    def __init__(self, max_shift=100):
        super().__init__()

        self.max_shift = max_shift

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        xy_shift = (torch.rand(2).numpy() - 0.5) * 2 * self.max_shift
        xy_shift *= meta.get('group_i', 1.0)

        # shift image
        affine_params = (1.0, 0.0, xy_shift[0], 0.0, 1.0, xy_shift[1])
        image = image.transform(image.size, PIL.Image.AFFINE, affine_params,
                                fillcolor=(127, 127, 127))

        # shift all annotations
        for ann in anns:
            ann['keypoints'][:, :2] += xy_shift
            ann['bbox'][:2] += xy_shift

        # adjust meta
        LOG.debug('meta before: %s', meta)
        meta['offset'] += xy_shift
        meta['valid_area'][:2] += xy_shift
        LOG.debug('meta after: %s', meta)

        return image, anns, meta
