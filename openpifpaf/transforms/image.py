import io
import logging

import numpy as np
import PIL
import torch

from .preprocess import Preprocess

try:
    import scipy
except ImportError:
    scipy = None

LOG = logging.getLogger(__name__)


class ImageTransform(Preprocess):
    """Transform image without modifying annotations or meta."""
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns, meta):
        image = self.image_transform(image)
        return image, anns, meta


class JpegCompression(Preprocess):
    """Add jpeg compression."""
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, image, anns, meta):
        f = io.BytesIO()
        image.save(f, 'jpeg', quality=self.quality)
        return PIL.Image.open(f), anns, meta


class Blur(Preprocess):
    """Blur image."""
    def __init__(self, max_sigma=5.0):
        self.max_sigma = max_sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.max_sigma * float(torch.rand(1).item())
        im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
        return PIL.Image.fromarray(im_np), anns, meta


class HorizontalBlur(Preprocess):
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.sigma * (0.8 + 0.4 * float(torch.rand(1).item()))
        LOG.debug('horizontal blur with %f', sigma)
        im_np = scipy.ndimage.filters.gaussian_filter1d(im_np, sigma=sigma, axis=1)
        return PIL.Image.fromarray(im_np), anns, meta
