"""Utilities for decoders."""

import functools
import numpy as np


@functools.lru_cache(maxsize=16)
def index_field(shape):
    yx = np.indices(shape, dtype=np.float32)
    xy = np.flip(yx, axis=0)
    return xy


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


def scalar_square_add_single(field, x, y, sigma, value):
    minx = max(0, int(x - sigma))
    miny = max(0, int(y - sigma))
    maxx = max(minx + 1, min(field.shape[1], int(x + sigma) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(y + sigma) + 1))
    field[miny:maxy, minx:maxx] += value
