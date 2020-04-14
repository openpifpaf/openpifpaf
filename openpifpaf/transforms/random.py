import torch

from .preprocess import Preprocess


class RandomApply(Preprocess):
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, anns, meta):
        if float(torch.rand(1).item()) > self.probability:
            return image, anns, meta
        return self.transform(image, anns, meta)


class DeterministicEqualChoice(Preprocess):
    def __init__(self, transforms, exponent=1):
        """Use a modular exponent hash to choose the transformation
        for a given image id.
        Use different exponents when using multiple choices.
        """
        self.transforms = transforms
        self.exponent = exponent

    def __call__(self, image, anns, meta):
        choice = pow(meta['image_id'], self.exponent, len(self.transforms))
        t = self.transforms[choice]
        return t(image, anns, meta)
