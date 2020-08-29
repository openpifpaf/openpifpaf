import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import scalar_square_add_gauss_with_max
from .. import visualizer

LOG = logging.getLogger(__name__)


class CifHr:
    neighbors = 16
    v_threshold = 0.1
    debug_visualizer = visualizer.CifHr()

    def __init__(self):
        self.accumulated = None

    def fill_single(self, cif, meta):
        return self.fill([cif], [meta])

    def accumulate(self, len_cifs, t, p, stride, min_scale):
        p = p[:, p[0] > self.v_threshold]
        if min_scale:
            p = p[:, p[4] > min_scale / stride]

        v, x, y, _, scale = p
        x = x * stride
        y = y * stride
        sigma = np.maximum(1.0, 0.5 * scale * stride)

        # Occupancy covers 2sigma.
        # Restrict this accumulation to 1sigma so that seeds for the same joint
        # are properly suppressed.
        scalar_square_add_gauss_with_max(
            t, x, y, sigma, v / self.neighbors / len_cifs, truncate=1.0)

    def fill(self, cifs, metas):
        start = time.perf_counter()

        if self.accumulated is None:
            shape = (
                cifs[0].shape[0],
                int((cifs[0].shape[2] - 1) * metas[0].stride + 1),
                int((cifs[0].shape[3] - 1) * metas[0].stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
        else:
            ta = np.zeros(self.accumulated.shape, dtype=np.float32)

        for cif, meta in zip(cifs, metas):
            for t, p in zip(ta, cif):
                self.accumulate(len(cifs), t, p, meta.stride, meta.decoder_min_scale)

        if self.accumulated is None:
            self.accumulated = ta
        else:
            self.accumulated = np.maximum(ta, self.accumulated)

        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        self.debug_visualizer.predicted(self.accumulated)
        return self


class CifDetHr(CifHr):
    def accumulate(self, len_cifs, t, p, stride, min_scale):
        p = p[:, p[0] > self.v_threshold]
        if min_scale:
            p = p[:, p[4] > min_scale / stride]
            p = p[:, p[5] > min_scale / stride]

        v, x, y, _, w, h, _ = p
        x = x * stride
        y = y * stride
        sigma = np.maximum(1.0, 0.1 * np.minimum(w, h) * stride)

        # Occupancy covers 2sigma.
        # Restrict this accumulation to 1sigma so that seeds for the same joint
        # are properly suppressed.
        scalar_square_add_gauss_with_max(
            t, x, y, sigma, v / self.neighbors / len_cifs, truncate=1.0)
