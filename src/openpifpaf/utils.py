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


def rotate_box(bbox, width, height, angle_degrees):
    """Input bounding box is of the form x, y, width, height."""

    cangle = math.cos(angle_degrees / 180.0 * math.pi)
    sangle = math.sin(angle_degrees / 180.0 * math.pi)

    four_corners = np.array([
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0], bbox[1] + bbox[3]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
    ])

    x_old = four_corners[:, 0].copy() - width / 2
    y_old = four_corners[:, 1].copy() - height / 2
    four_corners[:, 0] = width / 2 + cangle * x_old + sangle * y_old
    four_corners[:, 1] = height / 2 - sangle * x_old + cangle * y_old

    x = np.min(four_corners[:, 0])
    y = np.min(four_corners[:, 1])
    xmax = np.max(four_corners[:, 0])
    ymax = np.max(four_corners[:, 1])

    return np.array([x, y, xmax - x, ymax - y])
