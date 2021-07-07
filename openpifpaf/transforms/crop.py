import copy
import logging

import numpy as np
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class Crop(Preprocess):
    """Random cropping."""

    def __init__(self, long_edge, use_area_of_interest=True):
        self.long_edge = long_edge
        self.use_area_of_interest = use_area_of_interest

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)
        original_valid_area = meta['valid_area'].copy()

        image, anns, ltrb = self.crop(image, anns, meta['valid_area'])
        meta['offset'] += ltrb[:2]

        new_wh = image.size
        LOG.debug('valid area before crop of %s: %s', ltrb, original_valid_area)
        # process crops from left and top
        meta['valid_area'][:2] = np.maximum(0.0, original_valid_area[:2] - ltrb[:2])
        # process crops from right and bottom
        new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
        new_rb_corner = np.maximum(meta['valid_area'][:2], new_rb_corner)
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

        return image, anns, meta

    @staticmethod
    def area_of_interest(anns, valid_area):
        """area that contains annotations with keypoints"""

        points_of_interest = [
            xy
            for ann in anns
            if not ann.get('iscrowd', False)
            for xy in [ann['bbox'][:2], ann['bbox'][:2] + ann['bbox'][2:]]
        ]
        if not points_of_interest:
            return valid_area
        points_of_interest = np.stack(points_of_interest, axis=0)
        min_xy = np.min(points_of_interest, axis=0) - 50
        max_xy = np.max(points_of_interest, axis=0) + 50

        # Make sure to stay inside of valid area.
        left = np.clip(min_xy[0], valid_area[0], valid_area[0] + valid_area[2] - 1)
        top = np.clip(min_xy[1], valid_area[1], valid_area[1] + valid_area[3] - 1)
        right = np.clip(max_xy[0], left + 1, valid_area[0] + valid_area[2])
        bottom = np.clip(max_xy[1], top + 1, valid_area[1] + valid_area[3])

        return (left, top, right - left, bottom - top)

    @staticmethod
    def random_location_1d(image_length,
                           valid_min, valid_length,
                           interest_min, interest_length,
                           crop_length,
                           tail=0.1, shift=0.0, fix_inconsistent=True):
        if image_length <= crop_length:
            return 0

        if fix_inconsistent:
            # relevant for tracking with inconsistent image sizes
            # (e.g. with RandomizeOneFrame augmentation)
            valid_min = np.clip(valid_min, 0, image_length)
            valid_length = np.clip(valid_length, 0, image_length - valid_min)
            interest_min = np.clip(interest_min, 0, image_length)
            interest_length = np.clip(interest_length, 0, image_length - interest_min)

        sticky_rnd = -tail + 2 * tail * torch.rand((1,)).item()
        sticky_rnd = np.clip(sticky_rnd, 0.0, 1.0)

        if interest_length > crop_length:
            # crop within area of interest
            sticky_rnd = np.clip(sticky_rnd + shift / interest_length, 0.0, 1.0)
            offset = interest_min + (interest_length - crop_length) * sticky_rnd
            return int(offset)

        # from above: interest_length < crop_length
        min_v = interest_min + interest_length - crop_length
        max_v = interest_min

        if valid_length > crop_length:
            # clip to valid area
            min_v = max(min_v, valid_min)
            max_v = max(min_v, min(max_v, valid_min + valid_length - crop_length))
        elif image_length > crop_length:
            # clip to image
            min_v = max(min_v, 0)
            max_v = max(min_v, min(max_v, 0 + image_length - crop_length))

        # image constraint
        min_v = np.clip(min_v, 0, image_length - crop_length)
        max_v = np.clip(max_v, 0, image_length - crop_length)

        assert max_v >= min_v
        sticky_rnd = np.clip(sticky_rnd + shift / (max_v - min_v + 1e-3), 0.0, 1.0)
        offset = min_v + (max_v - min_v) * sticky_rnd
        return int(offset)

    def crop(self, image, anns, valid_area):
        if self.use_area_of_interest:
            area_of_interest = self.area_of_interest(anns, valid_area)
        else:
            area_of_interest = valid_area

        w, h = image.size
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            x_offset = self.random_location_1d(
                w - 1,
                valid_area[0], valid_area[2],
                area_of_interest[0], area_of_interest[2],
                self.long_edge,
            )
        if h > self.long_edge:
            y_offset = self.random_location_1d(
                h - 1,
                valid_area[1], valid_area[3],
                area_of_interest[1], area_of_interest[3],
                self.long_edge
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

        # crop keypoints
        for ann in anns:
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset

        return image, anns, np.array(ltrb)
