import functools
import math
import numpy as np


@functools.lru_cache(maxsize=64)
def create_sink(side):
    if side == 1:
        return np.zeros((2, 1, 1))

    sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=np.float)
    sink = np.stack((
        sink1d.reshape(1, -1).repeat(side, axis=0),
        sink1d.reshape(-1, 1).repeat(side, axis=1),
    ), axis=0)
    return sink


def anns_to_keypoint_sets(anns):
    """Ignore crowded annotations."""
    keypoint_sets = [ann['keypoints'] for ann in anns if not ann['iscrowd']]
    if not keypoint_sets:
        return np.zeros((0, 17, 3))  # TODO: remove hard coded class number

    return np.stack(keypoint_sets)


def anns_to_bg_mask(width_height, anns, include_annotated=True):
    """Create background mask taking crowded annotations into account."""
    mask = np.ones(width_height[::-1], dtype=np.bool)
    for ann in anns:
        if include_annotated and \
           not ann['iscrowd'] and \
           'keypoints' in ann and \
           np.any(ann['keypoints'][:, 2] > 0):
            continue

        if 'mask' not in ann:
            bb = ann['bbox'].copy()
            bb[2:] += bb[:2]  # convert width and height to x2 and y2
            bb[0] = np.clip(bb[0], 0, mask.shape[1] - 1)
            bb[1] = np.clip(bb[1], 0, mask.shape[0] - 1)
            bb[2] = np.clip(bb[2], 0, mask.shape[1] - 1)
            bb[3] = np.clip(bb[3], 0, mask.shape[0] - 1)
            bb = bb.astype(np.int)
            mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1] = 0
            continue
        mask[ann['mask']] = 0
    return mask


def mask_valid_area(intensities, valid_area):
    if valid_area is None:
        return intensities

    intensities[:, :int(valid_area[1]), :] = 0
    intensities[:, :, :int(valid_area[0])] = 0
    max_i = int(math.ceil(valid_area[1] + valid_area[3]))
    max_j = int(math.ceil(valid_area[0] + valid_area[2]))
    if max_i < intensities.shape[1]:
        intensities[:, max_i:, :] = 0
    if max_j < intensities.shape[2]:
        intensities[:, :, max_j:] = 0

    return intensities
