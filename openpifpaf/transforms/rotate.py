import copy
import logging
import math

import numpy as np
import PIL
import scipy
import torch

from .preprocess import Preprocess
from . import utils

LOG = logging.getLogger(__name__)


class RotateBy90(Preprocess):
    def __init__(self, angle_perturbation=0.0, fixed_angle=None):
        super().__init__()

        self.angle_perturbation = angle_perturbation
        self.fixed_angle = fixed_angle

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, h = image.size
        if self.fixed_angle is not None:
            angle = self.fixed_angle
        else:
            rnd1 = float(torch.rand(1).item())
            angle = int(rnd1 * 4.0) * 90.0
            sym_rnd2 = (float(torch.rand(1).item()) - 0.5) * 2.0
            angle += sym_rnd2 * self.angle_perturbation
        LOG.debug('rotation angle = %f', angle)
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
                im_np = scipy.ndimage.rotate(im_np, angle=angle, cval=127, reshape=False)
            image = PIL.Image.fromarray(im_np)
        LOG.debug('rotated by = %f degrees', angle)

        # rotate keypoints
        cangle = math.cos(angle / 180.0 * math.pi)
        sangle = math.sin(angle / 180.0 * math.pi)
        for ann in anns:
            xy = ann['keypoints'][:, :2]
            x_old = xy[:, 0].copy() - (w - 1)/2
            y_old = xy[:, 1].copy() - (h - 1)/2
            xy[:, 0] = (w - 1)/2 + cangle * x_old + sangle * y_old
            xy[:, 1] = (h - 1)/2 - sangle * x_old + cangle * y_old
            ann['bbox'] = utils.rotate_box(ann['bbox'], w - 1, h - 1, angle)

        LOG.debug('meta before: %s', meta)
        meta['valid_area'] = utils.rotate_box(meta['valid_area'], w - 1, h - 1, angle)
        # fix valid area to be inside original image dimensions
        original_valid_area = meta['valid_area'].copy()
        meta['valid_area'][0] = np.clip(meta['valid_area'][0], 0, w)
        meta['valid_area'][1] = np.clip(meta['valid_area'][1], 0, h)
        new_rb_corner = original_valid_area[:2] + original_valid_area[2:]
        new_rb_corner[0] = np.clip(new_rb_corner[0], 0, w)
        new_rb_corner[1] = np.clip(new_rb_corner[1], 0, h)
        meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
        LOG.debug('meta after: %s', meta)

        return image, anns, meta
