import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import scalar_square_add_gauss_with_max
from .field_config import FieldConfig
from .. import visualizer

LOG = logging.getLogger(__name__)


class CifHr:
    neighbors = 16
    v_threshold = 0.1
    debug_visualizer = visualizer.CifHr()

    def __init__(self, config: FieldConfig):
        self.config = config
        self.accumulated = None

    def fill_cif(self, cif, stride, min_scale=0.0):
        return self.fill_multiple([cif], stride, min_scale)

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

    def fill_multiple(self, cifs, stride, min_scale=0.0):
        start = time.perf_counter()

        if self.accumulated is None:
            shape = (
                cifs[0].shape[0],
                int((cifs[0].shape[2] - 1) * stride + 1),
                int((cifs[0].shape[3] - 1) * stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
        else:
            ta = np.zeros(self.accumulated.shape, dtype=np.float32)

        for cif in cifs:
            for t, p in zip(ta, cif):
                self.accumulate(len(cifs), t, p, stride, min_scale)

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

        scalar_square_add_gauss_with_max(
            t, x, y, sigma, v / self.neighbors / len_cifs, truncate=2.0)
