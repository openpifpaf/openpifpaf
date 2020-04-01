import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import scalar_square_add_gauss_with_max
from .field_config import FieldConfig

LOG = logging.getLogger(__name__)


class PifHr(object):
    neighbors = 16
    v_threshold = 0.1
    debug_visualizer = None

    def __init__(self, config: FieldConfig):
        self.config = config
        self.accumulated = None

    def fill_cif(self, cif, stride, min_scale=0.0):
        return self.fill_multiple([cif], stride, min_scale)

    def fill_multiple(self, pifs, stride, min_scale=0.0):
        start = time.perf_counter()

        if self.accumulated is None:
            shape = (
                pifs[0].shape[0],
                int((pifs[0].shape[2] - 1) * stride + 1),
                int((pifs[0].shape[3] - 1) * stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
        else:
            ta = np.zeros(self.accumulated.shape, dtype=np.float32)

        for pif in pifs:
            for t, p in zip(ta, pif):
                p = p[:, p[0] > self.v_threshold]
                if min_scale:
                    p = p[:, p[4] > min_scale / stride]

                v, x, y, _, scale = p
                x = x * stride
                y = y * stride
                sigma = np.maximum(1.0, 0.5 * scale * stride)

                scalar_square_add_gauss_with_max(
                    t, x, y, sigma, v / self.neighbors / len(pifs), truncate=2.0)

        if self.accumulated is None:
            self.accumulated = ta
        else:
            self.accumulated = np.maximum(ta, self.accumulated)

        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        return self

    def fill(self, fields):
        if len(self.config.cif_indices) == 10:
            for cif_i1, cif_i2, stride, min_scale in zip(self.config.cif_indices[:5],
                                                         self.config.cif_indices[5:],
                                                         self.config.cif_strides[:5],
                                                         self.config.cif_min_scales[:5]):
                self.fill_multiple([fields[cif_i1], fields[cif_i2]], stride, min_scale=min_scale)
        else:
            for cif_i, stride, min_scale in zip(self.config.cif_indices,
                                                self.config.cif_strides,
                                                self.config.cif_min_scales):
                self.fill_cif(fields[cif_i], stride, min_scale=min_scale)

        if self.debug_visualizer is not None:
            self.debug_visualizer.predicted(self.accumulated)
        return self
