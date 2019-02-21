import numpy as np

from .data import COCO_KEYPOINTS, HFLIP


def horizontal_swap(keypoints):
    target = np.zeros_like(keypoints)

    for source_i, xyv in enumerate(keypoints):
        source_name = COCO_KEYPOINTS[source_i]
        target_name = HFLIP.get(source_name)
        if target_name:
            target_i = COCO_KEYPOINTS.index(target_name)
        else:
            target_i = source_i
        target[target_i] = xyv

    return target
