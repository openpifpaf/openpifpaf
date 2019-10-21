import copy
import logging

import numpy as np
import torch

from .preprocess import Preprocess

LOG = logging.getLogger(__name__)


class Crop(Preprocess):
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
        # process cropps from right and bottom
        new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
        new_rb_corner = np.maximum(0.0, new_rb_corner)
        new_rb_corner = np.minimum(new_wh, new_rb_corner)
        meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
        LOG.debug('valid area after crop: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    @staticmethod
    def area_of_interest(anns, valid_area):
        """area that contains annotations with keypoints"""

        anns_of_interest = [
            ann for ann in anns
            if not ann.get('iscrowd', False) and np.any(ann['keypoints'][:, 2] > 0)
        ]
        if not anns_of_interest:
            return valid_area

        min_x = min(np.min(ann['keypoints'][ann['keypoints'][:, 2] > 0, 0])
                    for ann in anns_of_interest) - 50
        min_y = min(np.min(ann['keypoints'][ann['keypoints'][:, 2] > 0, 1])
                    for ann in anns_of_interest) - 50
        max_x = max(np.max(ann['keypoints'][ann['keypoints'][:, 2] > 0, 0])
                    for ann in anns_of_interest) + 50
        max_y = max(np.max(ann['keypoints'][ann['keypoints'][:, 2] > 0, 1])
                    for ann in anns_of_interest) + 50

        topleft = (
            max(valid_area[0], min_x),
            max(valid_area[1], min_y),
        )
        bottomright = (
            min(valid_area[0] + valid_area[2], max_x),
            min(valid_area[1] + valid_area[3], max_y),
        )

        return (
            topleft[0],
            topleft[1],
            bottomright[0] - topleft[0],
            bottomright[1] - topleft[1],
        )

    def crop(self, image, anns, valid_area):
        if self.use_area_of_interest:
            area_of_interest = self.area_of_interest(anns, valid_area)
        else:
            area_of_interest = valid_area

        w, h = image.size
        padding = int(self.long_edge / 2.0)
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            min_x = int(area_of_interest[0])
            max_x = int(area_of_interest[0] + area_of_interest[2]) - self.long_edge
            if max_x > min_x:
                x_offset = torch.randint(-padding + min_x, max_x + padding, (1,))
                x_offset = torch.clamp(x_offset, min=min_x, max=max_x).item()
            else:
                x_offset = min_x
        if h > self.long_edge:
            min_y = int(area_of_interest[1])
            max_y = int(area_of_interest[1] + area_of_interest[3]) - self.long_edge
            if max_y > min_y:
                y_offset = torch.randint(-padding + min_y, max_y + padding, (1,))
                y_offset = torch.clamp(y_offset, min=min_y, max=max_y).item()
            else:
                y_offset = min_y
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
