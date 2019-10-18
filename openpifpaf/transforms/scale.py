import copy
import logging

import numpy as np
import PIL
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


def _scale(image, anns, meta, target_w, target_h, resample):
    """target_w and target_h as integers"""
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)

    # scale image
    w, h = image.size
    image = image.resize((target_w, target_h), resample)
    LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)

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

    for ann in anns:
        ann['valid_area'] = meta['valid_area']

    return image, anns, meta


class RescaleRelative(Preprocess):
    def __init__(self, scale_range=(0.5, 1.0), *,
                 resample=PIL.Image.BICUBIC,
                 power_law=False):
        self.scale_range = scale_range
        self.resample = resample
        self.power_law = power_law

    def __call__(self, image, anns, meta):
        if isinstance(self.scale_range, tuple):
            if self.power_law:
                rnd_range = np.log2(self.scale_range[0]), np.log2(self.scale_range[1])
                log2_scale_factor = (
                    rnd_range[0] +
                    torch.rand(1).item() * (rnd_range[1] - rnd_range[0])
                )
                # mean = 0.5 * (rnd_range[0] + rnd_range[1])
                # sigma = 0.5 * (rnd_range[1] - rnd_range[0])
                # log2_scale_factor = mean + sigma * torch.randn(1).item()

                scale_factor = 2 ** log2_scale_factor
                # LOG.debug('mean = %f, sigma = %f, log2r = %f, scale = %f',
                #           mean, sigma, log2_scale_factor, scale_factor)
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
        target_w, target_h = int(w * scale_factor), int(h * scale_factor)
        return _scale(image, anns, meta, target_w, target_h, self.resample)


class RescaleAbsolute(Preprocess):
    def __init__(self, long_edge, *, resample=PIL.Image.BICUBIC):
        self.long_edge = long_edge
        self.resample = resample

    def __call__(self, image, anns, meta):
        w, h = image.size

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = int(torch.randint(this_long_edge[0], this_long_edge[1], (1,)).item())

        s = this_long_edge / max(h, w)
        if h > w:
            target_w, target_h = int(w * s), this_long_edge
        else:
            target_w, target_h = this_long_edge, int(h * s)
        return _scale(image, anns, meta, target_w, target_h, self.resample)
