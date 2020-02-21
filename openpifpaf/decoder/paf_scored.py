import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import scalar_values

LOG = logging.getLogger(__name__)


class PafScored(object):
    def __init__(self, pifhr, skeleton, *, score_th, pif_floor=0.1):
        self.pifhr = pifhr
        self.skeleton = skeleton
        self.score_th = score_th
        self.pif_floor = pif_floor

        self.forward = None
        self.backward = None

    def fill(self, paf, stride, min_distance=0.0, max_distance=None):
        start = time.perf_counter()

        if self.forward is None:
            self.forward = [np.empty((9, 0), dtype=paf.dtype) for _ in paf]
            self.backward = [np.empty((9, 0), dtype=paf.dtype) for _ in paf]

        for paf_i, nine in enumerate(paf):
            assert nine.shape[0] == 9

            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]

            if min_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist > min_distance / stride
                nine = nine[:, mask_dist]

            if max_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist < max_distance / stride
                nine = nine[:, mask_dist]

            nine = np.copy(nine)
            nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= stride
            scores = nine[0]

            j1i = self.skeleton[paf_i][0] - 1
            if self.pif_floor < 1.0:
                pifhr_b = scalar_values(self.pifhr[j1i], nine[1], nine[2], default=0.0)
                scores_b = scores * (self.pif_floor + (1.0 - self.pif_floor) * pifhr_b)
            else:
                scores_b = scores
            mask_b = scores_b > self.score_th
            d9_b = np.copy(nine[:, mask_b][(0, 5, 6, 7, 8, 1, 2, 3, 4), :])
            d9_b[0] = scores_b[mask_b]
            self.backward[paf_i] = np.concatenate((self.backward[paf_i], d9_b), axis=1)

            j2i = self.skeleton[paf_i][1] - 1
            if self.pif_floor < 1.0:
                pifhr_f = scalar_values(self.pifhr[j2i], nine[5], nine[6], default=0.0)
                scores_f = scores * (self.pif_floor + (1.0 - self.pif_floor) * pifhr_f)
            else:
                scores_f = scores
            mask_f = scores_f > self.score_th
            d9_f = np.copy(nine[:, mask_f])
            d9_f[0] = scores_f[mask_f]
            self.forward[paf_i] = np.concatenate((self.forward[paf_i], d9_f), axis=1)

        LOG.debug('scored paf (%d, %d) in %.3fs',
                  sum(f.shape[1] for f in self.forward),
                  sum(b.shape[1] for b in self.backward),
                  time.perf_counter() - start)
        return self

    def fill_sequence(self, pafs, strides, min_distances, max_distances):
        for paf, stride, min_distance, max_distance in zip(
                pafs, strides, min_distances, max_distances):
            self.fill(paf, stride, min_distance=min_distance, max_distance=max_distance)

        return self
