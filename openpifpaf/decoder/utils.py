"""Utilities for decoders."""

import numpy as np


def index_field(shape):
    yx = np.indices(shape, dtype=np.float)
    xy = np.flip(yx, axis=0)
    return xy


def weiszfeld_nd(x, init_y, weights=None, epsilon=1e-8, max_steps=20):
    """Weighted Weiszfeld step."""
    if weights is None:
        weights = np.ones(x.shape[0])
    weights = np.expand_dims(weights, -1)
    weights_x = weights * x

    y = init_y
    for _ in range(max_steps):
        prev_y = y

        denom = np.linalg.norm(x - prev_y, axis=-1, keepdims=True) + epsilon
        y = (
            np.sum(weights_x / denom, axis=0) /
            np.sum(weights / denom, axis=0)
        )
        if np.sum(np.abs(prev_y - y)) < 1e-2:
            return y, denom

    return y, denom


def sparse_bilinear_kernel(coord, value):
    l = coord.astype(int)
    g = np.meshgrid(*((ll, ll + 1) for ll in l))
    g = list(zip(*(gg.reshape(-1) for gg in g)))

    v = [np.prod(1.0 - np.abs(coord-corner)) * value for corner in g]
    return g, v


class Sparse2DGaussianField(object):
    def __init__(self, data=None, nearest_neighbors=25):
        if data is None:
            data = np.zeros((0, 3))

        self.nearest_neighbors = nearest_neighbors
        self.data = data

    def value(self, xy, sigma):
        mask = np.logical_and(
            np.logical_and(self.data[0] > xy[0] - 2*sigma,
                           self.data[0] < xy[0] + 2*sigma),
            np.logical_and(self.data[1] > xy[1] - 2*sigma,
                           self.data[1] < xy[1] + 2*sigma),
        )
        diff = np.expand_dims(xy, -1) - self.data[:2, mask]
        if diff.shape[1] == 0:
            return 0.0

        gauss_1d = np.exp(-0.5 * diff**2 / sigma**2)
        gauss = np.prod(gauss_1d, axis=0)

        v = np.sum(gauss * self.data[2, mask])
        return np.tanh(v * 3.0 / self.nearest_neighbors)

    def values(self, xys, sigmas):
        assert xys.shape[-1] == 2
        if xys.shape[0] == 0:
            return np.zeros((0,))

        if isinstance(sigmas, float):
            sigmas = np.full((xys.shape[0],), sigmas)
        if hasattr(sigmas, 'shape') and sigmas.shape[0] == 1 and xys.shape[0] > 1:
            sigmas = np.full((xys.shape[0],), sigmas[0])

        return np.stack([self.value(xy, sigma) for xy, sigma in zip(xys, sigmas)])


def normalize_paf(intensity_fields, j1_fields, j2_fields, j1_fields_logb, j2_fields_logb, *,
                  fixed_b=None):
    intensity_fields = np.expand_dims(intensity_fields, 1)
    j1_fields_b = np.expand_dims(np.exp(j1_fields_logb), 1)
    j2_fields_b = np.expand_dims(np.exp(j2_fields_logb), 1)
    if fixed_b:
        j1_fields_b = np.full_like(j1_fields_b, fixed_b)
        j2_fields_b = np.full_like(j2_fields_b, fixed_b)

    index_fields = index_field(j1_fields[0, 0].shape)
    index_fields = np.expand_dims(index_fields, 0)
    j1_fields3 = np.concatenate((intensity_fields, index_fields + j1_fields, j1_fields_b), axis=1)
    j2_fields3 = np.concatenate((intensity_fields, index_fields + j2_fields, j2_fields_b), axis=1)
    paf = np.stack((j1_fields3, j2_fields3), axis=1)

    return paf


def normalize_pif(joint_intensity_fields, joint_fields, _, scale_fields, *,
                  fixed_scale=None):
    joint_intensity_fields = np.expand_dims(joint_intensity_fields.copy(), 1)
    scale_fields = np.expand_dims(scale_fields, 1)
    if fixed_scale is not None:
        scale_fields[:] = fixed_scale

    index_fields = index_field(joint_fields.shape[-2:])
    index_fields = np.expand_dims(index_fields, 0)
    joint_fields = index_fields + joint_fields

    return np.concatenate(
        (joint_intensity_fields, joint_fields, scale_fields),
        axis=1,
    )


def normalize_pifs(joint_intensity_fields, joint_fields, scale_fields, *,
                   fixed_scale=None):
    joint_intensity_fields = np.expand_dims(joint_intensity_fields.copy(), 1)
    scale_fields = np.expand_dims(scale_fields, 1)
    if fixed_scale is not None:
        scale_fields[:] = fixed_scale

    index_fields = index_field(joint_fields.shape[-2:])
    index_fields = np.expand_dims(index_fields, 0)
    joint_fields = index_fields + joint_fields

    return np.concatenate(
        (joint_intensity_fields, joint_fields, scale_fields),
        axis=1,
    )


def scalar_square_add_single(field, x, y, width, value):
    minx = max(0, int(round(x - width)))
    miny = max(0, int(round(y - width)))
    maxx = max(minx + 1, min(field.shape[1], int(round(x + width)) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(round(y + width)) + 1))
    field[miny:maxy, minx:maxx] += value
