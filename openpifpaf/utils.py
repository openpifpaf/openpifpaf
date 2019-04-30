import math
import numpy as np

from .data import COCO_KEYPOINTS, HFLIP


def horizontal_swap_coco(keypoints):
    target = np.zeros(keypoints.shape)

    for source_i, xyv in enumerate(keypoints):
        source_name = COCO_KEYPOINTS[source_i]
        target_name = HFLIP.get(source_name)
        if target_name:
            target_i = COCO_KEYPOINTS.index(target_name)
        else:
            target_i = source_i
        target[target_i] = xyv

    return target


def mask_valid_image(image, valid_area):
    image[:, :int(valid_area[1]), :] = 0
    image[:, :, :int(valid_area[0])] = 0
    max_i = int(math.ceil(valid_area[1] + valid_area[3]))
    max_j = int(math.ceil(valid_area[0] + valid_area[2]))
    if max_i < image.shape[1]:
        image[:, max_i:, :] = 0
    if max_j < image.shape[2]:
        image[:, :, max_j:] = 0
