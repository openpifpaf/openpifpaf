import copy
import logging

from ..preprocess import Preprocess

LOG = logging.getLogger(__name__)


class ImageToTracking(Preprocess):
    def __call__(self, image, anns, meta):
        anns0 = copy.deepcopy(anns)
        anns1 = copy.deepcopy(anns)
        meta0 = copy.deepcopy(meta)
        meta1 = copy.deepcopy(meta)

        meta0['group_i'] = 0
        meta1['group_i'] = 1

        for ann_i, (ann0, ann1) in enumerate(zip(anns0, anns1)):
            ann0['track_id'] = ann_i
            ann1['track_id'] = ann_i

        return [image, image], [anns0, anns1], [meta0, meta1]
