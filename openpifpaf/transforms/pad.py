import copy
import math
import logging

import torch
import torchvision

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class CenterPad(Preprocess):
    """Pad to a square of given size."""

    def __init__(self, target_size: int):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        LOG.debug('valid area before pad: %s, image size = %s, target size = %s',
                  meta['valid_area'], image.size, self.target_size)
        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s, image size = %s', meta['valid_area'], image.size)

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size

        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        left = max(0, left)
        top = max(0, top)

        right = self.target_size[0] - w - left
        bottom = self.target_size[1] - h - top
        right = max(0, right)
        bottom = max(0, bottom)
        ltrb = (left, top, right, bottom)
        LOG.debug('pad with %s', ltrb)

        # pad image
        fill_value = int(torch.randint(0, 255, (1,)).item())
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(fill_value, fill_value, fill_value))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb


class CenterPadTight(Preprocess):
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        LOG.debug('valid area before pad: %s, image size = %s', meta['valid_area'], image.size)
        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s, image size = %s', meta['valid_area'], image.size)

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size
        target_width = math.ceil((w - 1) / self.multiple) * self.multiple + 1
        target_height = math.ceil((h - 1) / self.multiple) * self.multiple + 1

        left = int((target_width - w) / 2.0)
        top = int((target_height - h) / 2.0)
        left = max(0, left)
        top = max(0, top)

        right = target_width - w - left
        bottom = target_height - h - top
        right = max(0, right)
        bottom = max(0, bottom)
        ltrb = (left, top, right, bottom)
        LOG.debug('pad with %s', ltrb)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb


class SquarePad(Preprocess):
    def __call__(self, image, anns, meta):
        center_pad = CenterPad(max(image.size))
        return center_pad(image, anns, meta)
