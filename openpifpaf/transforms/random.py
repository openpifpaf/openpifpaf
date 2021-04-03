import itertools
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


class RandomChoice(Preprocess):
    def __init__(self, transforms, probabilities):
        if sum(probabilities) < 1.0 and len(transforms) == len(probabilities):
            transforms.append(None)
        self.transforms = transforms

        if len(transforms) == len(probabilities) + 1:
            probabilities.append(1.0 - sum(probabilities))
        assert sum(probabilities) == 1.0, [transforms, probabilities]
        assert len(transforms) == len(probabilities)
        self.probabilities = probabilities

    def __call__(self, image, anns, meta):
        rnd = float(torch.rand(1).item())
        for t, p_cumulative in zip(self.transforms, itertools.accumulate(self.probabilities)):
            if rnd > p_cumulative:
                continue

            if t is None:
                return image, anns, meta
            return t(image, anns, meta)

        raise Exception('not possible')


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
