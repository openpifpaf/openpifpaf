import io
import logging

import numpy as np
import PIL
import scipy
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class ImageTransform(Preprocess):
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns, meta):
        image = self.image_transform(image)
        return image, anns, meta


class JpegCompression(Preprocess):
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, image, anns, meta):
        f = io.BytesIO()
        image.save(f, 'jpeg', quality=self.quality)
        return PIL.Image.open(f), anns, meta


class Blur(Preprocess):
    def __init__(self, max_sigma=5.0):
        self.max_sigma = max_sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.max_sigma * float(torch.rand(1).item())
        im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
        return PIL.Image.fromarray(im_np), anns, meta
