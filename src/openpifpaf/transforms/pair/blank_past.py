import copy
import logging
import random

import PIL

from ..preprocess import Preprocess

LOG = logging.getLogger(__name__)


class BlankPast(Preprocess):
    def __call__(self, images, all_anns, metas):
        all_anns = copy.deepcopy(all_anns)
        metas = copy.deepcopy(metas)

        for i, _ in enumerate(images[1:], start=1):
            images[i] = PIL.Image.new('RGB', (320, 240), (127, 127, 127))

        for i, _ in enumerate(all_anns[1:], start=1):
            all_anns[i] = []

        for meta in metas[1:]:
            meta['image'] = {'frame_id': -1, 'file_name': 'blank'}
            assert 'annotations' not in meta

        return images, all_anns, metas


class PreviousPast(Preprocess):
    def __init__(self):
        self.previous_image = PIL.Image.new('RGB', (320, 240), (127, 127, 127))
        self.previous_meta = {'frame_id': -1, 'file_name': 'blank'}
        self.previous_annotations = []

    def __call__(self, images, all_anns, metas):
        all_anns = copy.deepcopy(all_anns)
        metas = copy.deepcopy(metas)

        LOG.debug('replacing %s with %s', metas[1], self.previous_meta)

        for i, _ in enumerate(images[1:], start=1):
            images[i] = self.previous_image

        for i, _ in enumerate(all_anns[1:], start=1):
            all_anns[i] = []  # TODO assumes previous image has nothing to do with current

        for meta in metas[1:]:
            meta['image'] = self.previous_meta  # why image?
            assert 'annotations' not in meta  # why would there be anns in meta?

        self.previous_image = images[0]
        self.previous_annotations = all_anns[0]
        self.previous_meta = metas[0]
        return images, all_anns, metas


class RandomizeOneFrame(Preprocess):
    def __init__(self):
        self.previous_image = None
        self.previous_meta = None
        self.previous_annotations = []

    def __call__(self, images, all_anns, metas):
        all_anns = copy.deepcopy(all_anns)
        metas = copy.deepcopy(metas)

        replace_index = random.randrange(0, len(metas))

        if self.previous_image is not None:
            # image
            images[replace_index] = self.previous_image

            # annotations
            all_anns[replace_index] = self.previous_annotations
            if self.previous_meta.get('annotation_file', 0) \
               == metas[replace_index].get('annotation_file', 1):
                pass
            else:
                for ann in all_anns[replace_index]:
                    ann['track_id'] += 10000

            # meta
            metas[replace_index] = self.previous_meta

        not_replaced_index = 0 if replace_index != 0 else 1
        self.previous_image = copy.deepcopy(images[not_replaced_index])
        self.previous_annotations = copy.deepcopy(all_anns[not_replaced_index])
        self.previous_meta = copy.deepcopy(metas[not_replaced_index])
        return images, all_anns, metas
