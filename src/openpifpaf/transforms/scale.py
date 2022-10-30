import copy
import logging
import warnings

import numpy as np
import PIL.Image
import torch

from .preprocess import Preprocess

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import scipy.ndimage
except ImportError:
    scipy = None  # pylint: disable=invalid-name

# For Pillow<9 compatibility
if not hasattr(PIL.Image, "Resampling"):
    PIL.Image.Resampling = PIL.Image

LOG = logging.getLogger(__name__)


def _scale(image, anns, meta, target_w, target_h, resample, *, fast=False):
    """target_w and target_h as integers

    Internally, resample in Pillow are aliases:
    PIL.Image.Resampling.BILINEAR = 2
    PIL.Image.Resampling.BICUBIC = 3
    """
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)
    w, h = image.size

    assert resample in (0, 2, 3)

    # scale image
    if fast and cv2 is not None:
        LOG.debug('using OpenCV for fast rescale')
        if resample == 0:
            cv_interpoltation = cv2.INTER_NEAREST
        elif resample == 2:
            cv_interpoltation = cv2.INTER_LINEAR
        elif resample == 3:
            cv_interpoltation = cv2.INTER_CUBIC
        else:
            raise NotImplementedError('resample of {} not implemented for OpenCV'.format(resample))
        im_np = np.asarray(image)
        im_np = cv2.resize(im_np, (target_w, target_h), interpolation=cv_interpoltation)
        image = PIL.Image.fromarray(im_np)
    elif fast:
        LOG.debug('Requested fast resizing without OpenCV. Using Pillow. '
                  'Install OpenCV for even faster image resizing.')
        image = image.resize((target_w, target_h), resample)
    else:
        order = resample
        if order == 2:
            order = 1

        im_np = np.asarray(image)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1), order=order)
        image = PIL.Image.fromarray(im_np)

    LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)
    assert image.size[0] == target_w
    assert image.size[1] == target_h

    # rescale keypoints
    x_scale = (image.size[0] - 1) / (w - 1)
    y_scale = (image.size[1] - 1) / (h - 1)
    scale_factors = np.array((x_scale, y_scale))
    for ann in anns:
        ann['keypoints'][:, [0, 1]] *= np.expand_dims(scale_factors, 0)
        ann['bbox'][:2] *= scale_factors
        ann['bbox'][2:] *= scale_factors

    # adjust meta
    LOG.debug('meta before: %s', meta)
    meta['offset'] *= scale_factors
    meta['scale'] *= scale_factors
    meta['valid_area'][:2] *= scale_factors
    meta['valid_area'][2:] *= scale_factors
    LOG.debug('meta after: %s', meta)

    return image, anns, meta


class RescaleRelative(Preprocess):
    """Rescale relative to input image."""

    def __init__(self, scale_range=(0.5, 1.0), *,
                 resample=PIL.Image.Resampling.BILINEAR,
                 absolute_reference=None,
                 fast=False,
                 power_law=False,
                 stretch_range=None):
        self.scale_range = scale_range
        self.resample = resample
        self.absolute_reference = absolute_reference
        self.fast = fast
        self.power_law = power_law
        self.stretch_range = stretch_range

    def __call__(self, image, anns, meta):
        if isinstance(self.scale_range, tuple):
            if self.power_law:
                rnd_range = np.log2(self.scale_range[0]), np.log2(self.scale_range[1])
                log2_scale_factor = (
                    rnd_range[0]
                    + torch.rand(1).item() * (rnd_range[1] - rnd_range[0])
                )

                scale_factor = 2 ** log2_scale_factor
                LOG.debug('rnd range = %s, log2_scale_Factor = %f, scale factor = %f',
                          rnd_range, log2_scale_factor, scale_factor)
            else:
                scale_factor = (
                    self.scale_range[0]
                    + torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
                )
        else:
            scale_factor = self.scale_range

        w, h = image.size
        if self.absolute_reference is not None:
            if w > h:
                h *= self.absolute_reference / w
                w = self.absolute_reference
            else:
                w *= self.absolute_reference / h
                h = self.absolute_reference

        stretch_factor = 1.0
        if self.stretch_range is not None:
            stretch_factor = (
                self.stretch_range[0]
                + torch.rand(1).item() * (self.stretch_range[1] - self.stretch_range[0])
            )

        target_w, target_h = int(w * scale_factor * stretch_factor), int(h * scale_factor)
        return _scale(image, anns, meta, target_w, target_h, self.resample, fast=self.fast)


class RescaleAbsolute(Preprocess):
    """Rescale to a given size."""

    def __init__(self, long_edge, *, fast=False, resample=PIL.Image.Resampling.BILINEAR):
        self.long_edge = long_edge
        self.fast = fast
        self.resample = resample

    def __call__(self, image, anns, meta):
        w, h = image.size

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = torch.randint(
                int(this_long_edge[0]),
                int(this_long_edge[1]), (1,)
            ).item()

        s = this_long_edge / max(h, w)
        if h > w:
            target_w, target_h = int(w * s), int(this_long_edge)
        else:
            target_w, target_h = int(this_long_edge), int(h * s)
        return _scale(image, anns, meta, target_w, target_h, self.resample, fast=self.fast)


class ScaleMix(Preprocess):
    def __init__(self, scale_threshold, *,
                 upscale_factor=2.0,
                 downscale_factor=0.5,
                 resample=PIL.Image.Resampling.BILINEAR):
        self.scale_threshold = scale_threshold
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor
        self.resample = resample

    def __call__(self, image, anns, meta):
        scales = np.array([
            np.sqrt(ann['bbox'][2] * ann['bbox'][3])
            for ann in anns if (not getattr(ann, 'iscrowd', False)
                                and np.any(ann['keypoints'][:, 2] > 0.0))
        ])
        LOG.debug('scale threshold = %f, scales = %s', self.scale_threshold, scales)
        if not scales.shape[0]:
            return image, anns, meta

        all_above_threshold = np.all(scales > self.scale_threshold)
        all_below_threshold = np.all(scales < self.scale_threshold)
        if not all_above_threshold and \
           not all_below_threshold:
            return image, anns, meta

        w, h = image.size
        if all_above_threshold:
            target_w, target_h = int(w / 2), int(h / 2)
        else:
            target_w, target_h = int(w * 2), int(h * 2)
        LOG.debug('scale mix from (%d, %d) to (%d, %d)', w, h, target_w, target_h)
        return _scale(image, anns, meta, target_w, target_h, self.resample)
