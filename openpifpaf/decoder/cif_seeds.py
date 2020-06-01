import logging
import time

# pylint: disable=import-error
from ..functional import scalar_values
from .field_config import FieldConfig
from .cif_hr import CifHr
from .. import visualizer

LOG = logging.getLogger(__name__)


class CifSeeds:
    threshold = None
    score_scale = 1.0
    debug_visualizer = visualizer.Seeds()

    def __init__(self, cifhr: CifHr, config: FieldConfig):
        self.cifhr = cifhr
        self.config = config
        self.seeds = []

    def fill_cif(self, cif, stride, *, min_scale=0.0, seed_mask=None):
        start = time.perf_counter()

        sv = 0.0

        for field_i, p in enumerate(cif):
            if seed_mask is not None and not seed_mask[field_i]:
                continue
            p = p[:, p[0] > self.threshold / 2.0]
            if min_scale:
                p = p[:, p[4] > min_scale / stride]
            _, x, y, _, s = p

            start_sv = time.perf_counter()
            v = scalar_values(self.cifhr[field_i], x * stride, y * stride)
            sv += time.perf_counter() - start_sv

            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold
            x, y, v, s = x[m] * stride, y[m] * stride, v[m], s[m] * stride

            for vv, xx, yy, ss in zip(v, x, y, s):
                self.seeds.append((vv, field_i, xx, yy, ss))

        LOG.debug('seeds %d, %.3fs (C++ %.3fs)', len(self.seeds), time.perf_counter() - start, sv)
        return self

    def get(self):
        self.debug_visualizer.predicted(self.seeds)
        return sorted(self.seeds, reverse=True)

    def fill(self, fields):
        for cif_i, stride, min_scale in zip(self.config.cif_indices,
                                            self.config.cif_strides,
                                            self.config.cif_min_scales):
            self.fill_cif(fields[cif_i], stride,
                          min_scale=min_scale,
                          seed_mask=self.config.seed_mask)

        return self


class CifDetSeeds(CifSeeds):
    def fill_cif(self, cif, stride, *, min_scale=0.0, seed_mask=None):
        start = time.perf_counter()

        for field_i, p in enumerate(cif):
            if seed_mask is not None and not seed_mask[field_i]:
                continue
            p = p[:, p[0] > self.threshold / 2.0]
            if min_scale:
                p = p[:, p[4] > min_scale / stride]
                p = p[:, p[5] > min_scale / stride]
            _, x, y, _, w, h, _ = p
            v = scalar_values(self.cifhr[field_i], x * stride, y * stride)
            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold
            x, y, v, w, h = x[m] * stride, y[m] * stride, v[m], w[m] * stride, h[m] * stride

            for vv, xx, yy, ww, hh in zip(v, x, y, w, h):
                self.seeds.append((vv, field_i, xx, yy, ww, hh))

        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self
