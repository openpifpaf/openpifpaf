import logging
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class RandomApply(Preprocess):
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, anns, meta):
        if float(torch.rand(1).item()) > self.probability:
            return image, anns, meta
        return self.transform(image, anns, meta)


class DeterministicEqualChoice(Preprocess):
    def __init__(self, transforms, salt=0):
        self.transforms = transforms
        self.salt = salt

    def __call__(self, image, anns, meta):
        assert meta['image_id'] > 0
        LOG.debug('image id = %d', meta['image_id'])
        choice = hash(meta['image_id'] + self.salt) % len(self.transforms)
        t = self.transforms[choice]
        if t is None:
            return image, anns, meta
        return t(image, anns, meta)
