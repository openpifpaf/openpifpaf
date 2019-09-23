import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import cumulative_average, scalar_square_add_gauss

LOG = logging.getLogger(__name__)


class PifHr(object):
    def __init__(self, pif_nn, v_th=0.1):
        self.pif_nn = pif_nn
        self.v_th = v_th

        self.target_accumulator = None
        self.scales = None
        self.scales_n = None

        self._clipped = None

    @property
    def targets(self):
        if self._clipped is not None:
            return self._clipped

        self._clipped = np.minimum(1.0, self.target_accumulator)
        return self._clipped

    def fill(self, pif, stride):
        start = time.perf_counter()

        if self.target_accumulator is None:
            shape = (
                pif.shape[0],
                int((pif.shape[2] - 1) * stride + 1),
                int((pif.shape[3] - 1) * stride + 1),
            )
            self.target_accumulator = np.zeros(shape, dtype=np.float32)
            self.scales = np.zeros(shape, dtype=np.float32)
            self.scales_n = np.zeros(shape, dtype=np.float32)

        for t, p, scale, n in zip(self.target_accumulator, pif, self.scales, self.scales_n):
            v, x, y, s = p[:, p[0] > self.v_th]
            x = x * stride
            y = y * stride
            s = s * stride
            scalar_square_add_gauss(t, x, y, s, v / self.pif_nn)
            cumulative_average(scale, n, x, y, s, s, v)

        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        return self
