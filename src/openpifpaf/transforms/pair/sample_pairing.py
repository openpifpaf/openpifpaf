import copy
import logging

import PIL

from ..preprocess import Preprocess

LOG = logging.getLogger(__name__)


class SamplePairing(Preprocess):
    """Sample Pairing

    Implements:
    @article{inoue2018data,
        title={Data augmentation by pairing samples for images classification},
        author={Inoue, Hiroshi},
        journal={arXiv preprint arXiv:1801.02929},
        year={2018}
    }
    This was originally published for classification and adapted here for
    pose estimation.
    """

    def __init__(self):
        self.previous_images = None
        self.previous_all_annotations = []

    def __call__(self, original_images, original_all_anns, metas):
        images = original_images
        all_anns = copy.deepcopy(original_all_anns)

        if self.previous_images is not None:
            # image
            images = [
                PIL.Image.blend(current_image, previous_image, 0.5)
                for current_image, previous_image in zip(images, self.previous_images)
            ]

            # annotations
            for current_anns, previous_anns in zip(all_anns, self.previous_all_annotations):
                current_anns += previous_anns

            # meta untouched

        self.previous_images = original_images
        self.previous_all_annotations = original_all_anns
        return images, all_anns, metas
