import functools
import math
import numpy as np


@functools.lru_cache(maxsize=64)
def create_sink(side):
    if side == 1:
        return np.zeros((2, 1, 1))

    sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=np.float32)
    sink = np.stack((
        sink1d.reshape(1, -1).repeat(side, axis=0),
        sink1d.reshape(-1, 1).repeat(side, axis=1),
    ), axis=0)
    return sink


def mask_valid_area(intensities, valid_area, *, fill_value=0):
    """Mask area.

    Intensities is either a feature map or an image.
    """
    if valid_area is None:
        return

    if valid_area[1] >= 1.0:
        intensities[:, :int(valid_area[1]), :] = fill_value
    if valid_area[0] >= 1.0:
        intensities[:, :, :int(valid_area[0])] = fill_value

    max_i = int(math.ceil(valid_area[1] + valid_area[3])) + 1
    max_j = int(math.ceil(valid_area[0] + valid_area[2])) + 1
    if 0 < max_i < intensities.shape[1]:
        intensities[:, max_i:, :] = fill_value
    if 0 < max_j < intensities.shape[2]:
        intensities[:, :, max_j:] = fill_value
