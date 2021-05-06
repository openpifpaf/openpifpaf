import copy
import logging

import torch
import torchvision

import openpifpaf

LOG = logging.getLogger(__name__)


class Pad(openpifpaf.transforms.Preprocess):
    def __init__(self, target_size, max_shift):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.max_shift = max_shift

    def __call__(self, images, all_anns, metas):
        metas = copy.deepcopy(metas)
        all_anns = copy.deepcopy(all_anns)

        cam_shift = (torch.rand(2) - 0.5) * 2.0 * self.max_shift
        LOG.debug('max shift = %s, this shift = %s', self.max_shift, cam_shift)
        for meta_i, meta in enumerate(metas):
            LOG.debug('valid area before pad: %s, image size = %s',
                      meta['valid_area'], images[meta_i].size)
            images[meta_i], all_anns[meta_i], ltrb = self.center_pad(
                images[meta_i], all_anns[meta_i], cam_shift * meta.get('group_i', 1.0))
            meta['offset'] -= ltrb[:2]
            meta['valid_area'][:2] += ltrb[:2]
            LOG.debug('valid area after pad: %s, image size = %s',
                      meta['valid_area'], images[meta_i].size)

        return images, all_anns, metas

    def center_pad(self, image, anns, cam_shift):
        w, h = image.size

        left = (self.target_size[0] - w) / 2.0 + cam_shift[0]
        top = (self.target_size[1] - h) / 2.0 + cam_shift[1]
        left = int(torch.clamp(left, 0, self.target_size[0] - w).item())
        top = int(torch.clamp(top, 0, self.target_size[1] - h).item())

        right = self.target_size[0] - w - left
        bottom = self.target_size[1] - h - top
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
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
