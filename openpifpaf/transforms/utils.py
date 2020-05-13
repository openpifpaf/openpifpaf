import math
import numpy as np


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

    x_old = four_corners[:, 0].copy() - width/2
    y_old = four_corners[:, 1].copy() - height/2
    four_corners[:, 0] = width/2 + cangle * x_old + sangle * y_old
    four_corners[:, 1] = height/2 - sangle * x_old + cangle * y_old

    x = np.min(four_corners[:, 0])
    y = np.min(four_corners[:, 1])
    xmax = np.max(four_corners[:, 0])
    ymax = np.max(four_corners[:, 1])

    return np.array([x, y, xmax - x, ymax - y])
