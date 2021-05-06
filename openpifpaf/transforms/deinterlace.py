import copy
import logging

import numpy as np
import PIL

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class Deinterlace(Preprocess):
    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        # deinterlace image
        w, h = image.size
        image = PIL.Image.fromarray(np.asarray(image)[::2, ::2])
        LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)

        # rescale keypoints
        for ann in anns:
            ann['keypoints'][:, :2] *= 0.5
            ann['bbox'] *= 0.5

        LOG.debug('meta before: %s', meta)
        meta['offset'] *= 0.5
        meta['scale'] *= 0.5
        meta['valid_area'] *= 0.5
        LOG.debug('meta after: %s', meta)

        return image, anns, meta
