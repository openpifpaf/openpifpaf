import copy
import logging
import warnings

import numpy as np
import PIL
import scipy.ndimage
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


def _scale(image, anns, meta, target_w, target_h, resample, *, fast=False):
    """target_w and target_h as integers

    Internally, resample in Pillow are aliases:
    PIL.Image.BILINEAR = 2
    PIL.Image.BICUBIC = 3
    """
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)
    w, h = image.size

    assert resample in (0, 2, 3)

    # scale image
    if fast:
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
    for ann in anns:
        ann['keypoints'][:, 0] = ann['keypoints'][:, 0] * x_scale
        ann['keypoints'][:, 1] = ann['keypoints'][:, 1] * y_scale
        ann['bbox'][0] *= x_scale
        ann['bbox'][1] *= y_scale
        ann['bbox'][2] *= x_scale
        ann['bbox'][3] *= y_scale

    # adjust meta
    scale_factors = np.array((x_scale, y_scale))
    LOG.debug('meta before: %s', meta)
    meta['offset'] *= scale_factors
    meta['scale'] *= scale_factors
    meta['valid_area'][:2] *= scale_factors
    meta['valid_area'][2:] *= scale_factors
    LOG.debug('meta after: %s', meta)

    return image, anns, meta


class RescaleRelative(Preprocess):
    def __init__(self, scale_range=(0.5, 1.0), *,
                 resample=PIL.Image.BILINEAR,
                 absolute_reference=None,
                 fast=False,
                 power_law=False):
        self.scale_range = scale_range
        self.resample = resample
        self.absolute_reference = absolute_reference
        self.fast = fast
        self.power_law = power_law

    def __call__(self, image, anns, meta):
        if isinstance(self.scale_range, tuple):
            if self.power_law:
                rnd_range = np.log2(self.scale_range[0]), np.log2(self.scale_range[1])
                log2_scale_factor = (
                    rnd_range[0] +
                    torch.rand(1).item() * (rnd_range[1] - rnd_range[0])
                )

                scale_factor = 2 ** log2_scale_factor
                LOG.debug('rnd range = %s, log2_scale_Factor = %f, scale factor = %f',
                          rnd_range, log2_scale_factor, scale_factor)
            else:
                scale_factor = (
                    self.scale_range[0] +
                    torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
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
        target_w, target_h = int(w * scale_factor), int(h * scale_factor)
        return _scale(image, anns, meta, target_w, target_h, self.resample, fast=self.fast)


class RescaleAbsolute(Preprocess):
    def __init__(self, long_edge, *, fast=False, resample=PIL.Image.BILINEAR):
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
                 resample=PIL.Image.BILINEAR):
        self.scale_threshold = scale_threshold
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor
        self.resample = resample

    def __call__(self, image, anns, meta):
        scales = np.array([
            np.sqrt(ann['bbox'][2] * ann['bbox'][3])
            for ann in anns if (not getattr(ann, 'iscrowd', False) and
                                np.any(ann['keypoints'][:, 2] > 0.0))
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
