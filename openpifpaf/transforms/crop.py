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
    def area_of_interest(anns, valid_area, edge_length):
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

        # Make sure to stay inside of valid area.
        # Also make sure that the remaining window inside the valid area
        # has at least an edge_length in size (for the case where there is
        # only a small annotation in a corner).
        valid_area_r = valid_area[0] + valid_area[2]
        valid_area_b = valid_area[1] + valid_area[3]
        range_x = max(0, valid_area[2] - edge_length)
        range_y = max(0, valid_area[3] - edge_length)
        left = np.clip(min_x, valid_area[0], valid_area[0] + range_x)
        top = np.clip(min_y, valid_area[1], valid_area[1] + range_y)
        right = np.clip(max_x, valid_area_r - range_x, valid_area_r)
        bottom = np.clip(max_y, valid_area_b - range_y, valid_area_b)

        return (left, top, right - left, bottom - top)

    def crop(self, image, anns, valid_area):
        if self.use_area_of_interest:
            area_of_interest = self.area_of_interest(anns, valid_area, self.long_edge)
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
