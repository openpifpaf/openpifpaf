import copy
import logging
import math

import numpy as np
import PIL
import torch

from .pad import CenterPad
from .preprocess import Preprocess
from .. import utils

try:
    import scipy
except ImportError:
    scipy = None

LOG = logging.getLogger(__name__)


def rotate(image, anns, meta, angle):
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)

    LOG.debug('rotation angle = %f', angle)
    w, h = image.size
    assert meta['rotation']['angle'] == 0.0
    meta['rotation']['angle'] = angle
    meta['rotation']['width'] = w
    meta['rotation']['height'] = h

    # rotate image
    if angle != 0.0:
        im_np = np.asarray(image)
        if im_np.shape[0] == im_np.shape[1] and angle == 90:
            im_np = np.swapaxes(im_np, 0, 1)
            im_np = np.flip(im_np, axis=0)
        elif im_np.shape[0] == im_np.shape[1] and angle == 270:
            im_np = np.swapaxes(im_np, 0, 1)
            im_np = np.flip(im_np, axis=1)
        elif im_np.shape[0] == im_np.shape[1] and angle == 180:
            im_np = np.flip(im_np, axis=0)
            im_np = np.flip(im_np, axis=1)
        else:
            fill_value = int(torch.randint(0, 255, (1,)).item())
            im_np = scipy.ndimage.rotate(im_np, angle=angle, cval=fill_value, reshape=False)
        image = PIL.Image.fromarray(im_np)
    LOG.debug('rotated by = %f degrees', angle)

    # rotate keypoints
    cangle = math.cos(angle / 180.0 * math.pi)
    sangle = math.sin(angle / 180.0 * math.pi)
    for ann in anns:
        xy = ann['keypoints'][:, :2]
        x_old = xy[:, 0].copy() - (w - 1) / 2
        y_old = xy[:, 1].copy() - (h - 1) / 2
        xy[:, 0] = (w - 1) / 2 + cangle * x_old + sangle * y_old
        xy[:, 1] = (h - 1) / 2 - sangle * x_old + cangle * y_old
        ann['bbox'] = utils.rotate_box(ann['bbox'], w - 1, h - 1, angle)

    LOG.debug('meta before: %s', meta)
    meta['valid_area'] = utils.rotate_box(meta['valid_area'], w - 1, h - 1, angle)
    # fix valid area to be inside original image dimensions
    original_valid_area = meta['valid_area'].copy()
    meta['valid_area'][0] = np.clip(meta['valid_area'][0], 0, w - 1)
    meta['valid_area'][1] = np.clip(meta['valid_area'][1], 0, h - 1)
    new_rb_corner = original_valid_area[:2] + original_valid_area[2:]
    new_rb_corner[0] = np.clip(new_rb_corner[0], 0, w - 1)
    new_rb_corner[1] = np.clip(new_rb_corner[1], 0, h - 1)
    meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
    LOG.debug('meta after: %s', meta)

    return image, anns, meta


def _prepad(image, anns, meta, angle):
    if abs(angle) < 0.3:
        return image, anns, meta

    w, h = image.size
    cos_angle = math.cos(abs(angle) * math.pi / 180.0)
    sin_angle = math.sin(abs(angle) * math.pi / 180.0)
    LOG.debug('angle = %f, cos = %f, sin = %f', angle, cos_angle, sin_angle)
    padded_size = (
        int(w * cos_angle + h * sin_angle) + 1,
        int(h * cos_angle + w * sin_angle) + 1,
    )
    center_pad = CenterPad(padded_size)
    return center_pad(image, anns, meta)


class RotateBy90(Preprocess):
    """Randomly rotate by multiples of 90 degrees."""

    def __init__(self, angle_perturbation=0.0, fixed_angle=None, prepad=False):
        super().__init__()

        self.angle_perturbation = angle_perturbation
        self.fixed_angle = fixed_angle
        self.prepad = prepad

    def __call__(self, image, anns, meta):
        if self.fixed_angle is not None:
            angle = self.fixed_angle
        else:
            rnd1 = float(torch.rand(1).item())
            angle = int(rnd1 * 4.0) * 90.0
            sym_rnd2 = (float(torch.rand(1).item()) - 0.5) * 2.0
            angle += sym_rnd2 * self.angle_perturbation

        if self.prepad:
            image, anns, meta = _prepad(image, anns, meta, angle)
        return rotate(image, anns, meta, angle)


class RotateUniform(Preprocess):
    """Rotate by a random angle uniformly drawn from a given angle range."""

    def __init__(self, max_angle=30.0, prepad=True):
        super().__init__()
        self.max_angle = max_angle
        self.prepad = prepad

    def __call__(self, image, anns, meta):
        sym_rnd = (float(torch.rand(1).item()) - 0.5) * 2.0
        angle = sym_rnd * self.max_angle

        if self.prepad:
            image, anns, meta = _prepad(image, anns, meta, angle)
        return rotate(image, anns, meta, angle)
