import copy
import logging

import numpy as np
import torch

from ..preprocess import Preprocess
from ..crop import Crop as SingleImageCrop

LOG = logging.getLogger(__name__)


class Crop(Preprocess):
    def __init__(self, long_edge, *, use_area_of_interest=True, max_shift=0):
        self.long_edge = long_edge
        self.use_area_of_interest = use_area_of_interest
        self.max_shift = max_shift

    def __call__(self, images, all_anns, metas):
        metas = copy.deepcopy(metas)
        all_anns = copy.deepcopy(all_anns)

        if self.use_area_of_interest:
            # crop around the interesting area in the past frame to
            # train whether the pose continues or not (unless the past
            # frame is blank)
            if all_anns[0] and not all_anns[1]:
                valid_area = metas[0]['valid_area']
                area_of_interest = SingleImageCrop.area_of_interest(
                    all_anns[0], valid_area)
            else:
                valid_area = metas[1]['valid_area']
                area_of_interest = SingleImageCrop.area_of_interest(
                    all_anns[1], valid_area)
        else:
            valid_area = metas[0]['valid_area']
            area_of_interest = valid_area

        new_images = []
        new_anns = []
        new_metas = []

        cam_shift = (torch.rand(2).numpy() - 0.5) * 2.0 * self.max_shift
        LOG.debug('max shift = %s, this shift = %s', self.max_shift, cam_shift)
        for image, anns, meta in zip(images, all_anns, metas):
            LOG.debug('meta group = %d', meta['group_i'])
            original_valid_area = meta['valid_area'].copy()
            with torch.random.fork_rng(devices=[]):
                image, anns, ltrb = self.crop(
                    image, anns, valid_area, area_of_interest, cam_shift * meta.get('group_i', 1.0))
                meta['offset'] += ltrb[:2]

                new_wh = image.size
                LOG.debug('valid area before crop of %s: %s', ltrb, original_valid_area)
                # process crops from left and top
                meta['valid_area'][:2] = np.maximum(0.0, original_valid_area[:2] - ltrb[:2])
                # process crops from right and bottom
                new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
                new_rb_corner = np.maximum(0.0, new_rb_corner)
                new_rb_corner = np.minimum(new_wh, new_rb_corner)
                meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
                LOG.debug('valid area after crop: %s', meta['valid_area'])

                # clip bounding boxes
                for ann in anns:
                    unclipped_bbox = ann['bbox'].copy()
                    ann['bbox'][:2] = np.maximum(meta['valid_area'][:2], ann['bbox'][:2])
                    new_rb = unclipped_bbox[:2] + unclipped_bbox[2:]
                    new_rb = np.maximum(ann['bbox'][:2], new_rb)
                    new_rb = np.minimum(meta['valid_area'][:2] + meta['valid_area'][2:], new_rb)
                    ann['bbox'][2:] = new_rb - ann['bbox'][:2]
                anns = [ann for ann in anns if ann['bbox'][2] > 0.0 and ann['bbox'][3] > 0.0]

            new_images.append(image)
            new_anns.append(anns)
            new_metas.append(meta)

        return new_images, new_anns, new_metas

    def crop(self, image, anns, valid_area, area_of_interest, cam_shift):
        LOG.debug('cam shift = %s', cam_shift)
        w, h = image.size
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            x_offset = SingleImageCrop.random_location_1d(
                w - 1,
                valid_area[0], valid_area[2],
                area_of_interest[0], area_of_interest[2],
                self.long_edge,
                shift=cam_shift[0],
                fix_inconsistent=True,
            )
        if h > self.long_edge:
            y_offset = SingleImageCrop.random_location_1d(
                h - 1,
                valid_area[1], valid_area[3],
                area_of_interest[1], area_of_interest[3],
                self.long_edge,
                shift=cam_shift[1],
                fix_inconsistent=True,
            )
        LOG.debug('crop offsets (%d, %d)', x_offset, y_offset)

        # crop image
        new_w = min(self.long_edge, w - x_offset)
        new_h = min(self.long_edge, h - y_offset)
        # ltrb might be confusing name:
        # it's the coordinates of the top-left corner and the coordinates
        # of the bottom right corner
        ltrb = (x_offset, y_offset, x_offset + new_w, y_offset + new_h)
        image = image.crop(ltrb)
        LOG.debug('cropped image: %s', ltrb)

        # crop keypoints
        for ann in anns:
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset

        return image, anns, np.array(ltrb)
