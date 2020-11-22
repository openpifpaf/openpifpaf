import logging
import numpy as np

from ...functional import scalar_nonzero_clipped_with_reduction

LOG = logging.getLogger(__name__)


def scalar_square_add_single(field, x, y, sigma, value):
    minx = max(0, int(x - sigma))
    miny = max(0, int(y - sigma))
    # +2: +1 for rounding up and another +1 for non-inclusive boundary
    maxx = max(minx + 1, min(field.shape[1], int(x + sigma) + 2))
    maxy = max(miny + 1, min(field.shape[0], int(y + sigma) + 2))
    field[miny:maxy, minx:maxx] += value


class Occupancy():
    def __init__(self, shape, reduction, *, min_scale=None):
        assert len(shape) == 3
        if min_scale is None:
            min_scale = reduction
        assert min_scale >= reduction

        self.reduction = reduction
        self.min_scale = min_scale
        self.min_scale_reduced = min_scale / reduction

        self.occupancy = np.zeros((
            shape[0],
            int(shape[1] / reduction) + 1,
            int(shape[2] / reduction) + 1,
        ), dtype=np.uint8)
        LOG.debug('shape = %s, min_scale = %d', self.occupancy.shape, self.min_scale_reduced)

    def __len__(self):
        return len(self.occupancy)

    def set(self, f, x, y, sigma):
        """Setting needs to be centered at the rounded (x, y)."""
        if f >= len(self.occupancy):
            return

        xi = round(x / self.reduction)
        yi = round(y / self.reduction)
        si = round(max(self.min_scale_reduced, sigma / self.reduction))
        scalar_square_add_single(self.occupancy[f], xi, yi, si, 1)

    def get(self, f, x, y):
        """Getting needs to be done at the floor of (x, y)."""
        if f >= len(self.occupancy):
            return 1.0

        # floor is done in scalar_nonzero_clipped below
        return scalar_nonzero_clipped_with_reduction(self.occupancy[f], x, y, self.reduction)
