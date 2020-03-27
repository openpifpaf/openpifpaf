import logging
import time

# pylint: disable=import-error
from ..functional import scalar_values
from .field_config import FieldConfig
from .pif_hr import PifHr

LOG = logging.getLogger(__name__)


class PifSeeds:
    threshold = None
    score_scale = 1.0
    debug_visualizer = None

    def __init__(self, pifhr: PifHr, config: FieldConfig):
        self.pifhr = pifhr
        self.config = config
        self.seeds = []

    def fill_cif(self, cif, stride, *, min_scale=0.0, seed_mask=None):
        start = time.perf_counter()

        for field_i, p in enumerate(cif):
            if seed_mask is not None and not seed_mask[field_i]:
                continue
            p = p[:, p[0] > self.threshold / 2.0]
            if min_scale:
                p = p[:, p[4] > min_scale / stride]
            _, x, y, _, s = p
            v = scalar_values(self.pifhr[field_i], x * stride, y * stride)
            m = v > self.threshold
            x, y, v, s = x[m] * stride, y[m] * stride, v[m], s[m] * stride

            for vv, xx, yy, ss in zip(v, x, y, s):
                self.seeds.append((vv, field_i, xx, yy, ss))

        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self

    def get(self):
        if self.debug_visualizer:
            self.debug_visualizer.seeds(self.seeds)

        seeds = self.seeds
        if self.score_scale != 1.0:
            seeds = [(self.score_scale * vv, ff, xx, yy, ss)
                     for vv, ff, xx, yy, ss in seeds]

        return sorted(self.seeds, reverse=True)

    def fill(self, fields):
        for cif_i, stride, min_scale in zip(self.config.cif_indices,
                                            self.config.cif_strides,
                                            self.config.cif_min_scales):
            self.fill_cif(fields[cif_i], stride,
                          min_scale=min_scale,
                          seed_mask=self.config.seed_mask)

        return self
