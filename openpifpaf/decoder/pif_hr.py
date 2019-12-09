import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import cumulative_average, scalar_square_add_gauss_with_max

LOG = logging.getLogger(__name__)


class PifHr(object):
    v_threshold = 0.1

    def __init__(self, pif_nn):
        self.pif_nn = pif_nn

        self.target_accumulator = None
        self.scales = None
        self.scales_n = None

        self._clipped = None

    @property
    def targets(self):
        return self.target_accumulator

    def fill(self, pif, stride, min_scale=0.0):
        return self.fill_multiple([pif], stride, min_scale)

    def fill_multiple(self, pifs, stride, min_scale=0.0):
        start = time.perf_counter()

        if self.target_accumulator is None:
            shape = (
                pifs[0].shape[0],
                int((pifs[0].shape[2] - 1) * stride + 1),
                int((pifs[0].shape[3] - 1) * stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
            self.scales = np.zeros(shape, dtype=np.float32)
            self.scales_n = np.zeros(shape, dtype=np.float32)
        else:
            ta = np.zeros(self.target_accumulator.shape, dtype=np.float32)

        for pif in pifs:
            for t, p, scale, n in zip(ta, pif, self.scales, self.scales_n):
                p = p[:, p[0] > self.v_threshold]
                if min_scale:
                    p = p[:, p[3] > min_scale / stride]

                v, x, y, s = p
                x = x * stride
                y = y * stride
                s = s * stride

                scalar_square_add_gauss_with_max(
                    t, x, y, s, v / self.pif_nn / len(pifs), truncate=1.0)
                cumulative_average(scale, n, x, y, s, s, v)

        if self.target_accumulator is None:
            self.target_accumulator = ta
        else:
            self.target_accumulator = np.maximum(ta, self.target_accumulator)

        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        return self

    def fill_sequence(self, pifs, strides, min_scales):
        if len(pifs) == 10:
            for pif1, pif2, stride, min_scale in zip(pifs[:5], pifs[5:], strides, min_scales):
                self.fill_multiple([pif1, pif2], stride, min_scale=min_scale)
        else:
            for pif, stride, min_scale in zip(pifs, strides, min_scales):
                self.fill(pif, stride, min_scale=min_scale)

        return self
