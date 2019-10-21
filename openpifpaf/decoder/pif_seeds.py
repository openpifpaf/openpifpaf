import logging
import time

# pylint: disable=import-error
from ..functional import scalar_values

LOG = logging.getLogger(__name__)


class PifSeeds(object):
    def __init__(self, pifhr, seed_threshold, *,
                 debug_visualizer=None):
        self.pifhr = pifhr
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.seeds = []

    def fill(self, pif, stride, min_scale=0.0):
        start = time.perf_counter()

        for field_i, p in enumerate(pif):
            p = p[:, p[0] > self.seed_threshold / 2.0]
            if min_scale:
                p = p[:, p[3] > min_scale / stride]
            _, x, y, s = p
            v = scalar_values(self.pifhr[field_i], x * stride, y * stride)
            m = v > self.seed_threshold
            x, y, v, s = x[m] * stride, y[m] * stride, v[m], s[m] * stride

            for vv, xx, yy, ss in zip(v, x, y, s):
                self.seeds.append((vv, field_i, xx, yy, ss))

        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self

    def get(self):
        if self.debug_visualizer:
            self.debug_visualizer.seeds(self.seeds)

        return sorted(self.seeds, reverse=True)

    def fill_sequence(self, pifs, strides, min_scales):
        for pif, stride, min_scale in zip(pifs, strides, min_scales):
            self.fill(pif, stride, min_scale=min_scale)

        return self
