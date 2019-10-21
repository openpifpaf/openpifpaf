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
            self.forward = [None for _ in paf]
            self.backward = [None for _ in paf]

        for paf_i, fourds in enumerate(paf):
            assert fourds.shape[0] == 2
            assert fourds.shape[1] == 4

            scores = np.min(fourds[:, 0], axis=0)
            mask = scores > self.score_th
            scores = scores[mask]
            fourds = fourds[:, :, mask]

            if min_distance:
                dist = np.linalg.norm(fourds[0, 1:3] - fourds[1, 1:3], axis=0)
                mask_dist = dist > min_distance / stride
                scores = scores[mask_dist]
                fourds = fourds[:, :, mask_dist]

            if max_distance:
                dist = np.linalg.norm(fourds[0, 1:3] - fourds[1, 1:3], axis=0)
                mask_dist = dist < max_distance / stride
                scores = scores[mask_dist]
                fourds = fourds[:, :, mask_dist]

            fourds = np.copy(fourds)
            fourds[:, 1] *= stride
            fourds[:, 2] *= stride
            fourds[:, 3] *= stride

            j1i = self.skeleton[paf_i][0] - 1
            if self.pif_floor < 1.0:
                pifhr_b = scalar_values(
                    self.pifhr[j1i], fourds[0, 1], fourds[0, 2], default=0.0)
                scores_b = scores * (self.pif_floor + (1.0 - self.pif_floor) * pifhr_b)
            else:
                scores_b = scores
            mask_b = scores_b > self.score_th
            d7_b = np.concatenate((
                np.expand_dims(scores_b[mask_b], 0),
                fourds[1, 1:4][:, mask_b],
                fourds[0, 1:4][:, mask_b],
            ))
            if self.backward[paf_i] is None:
                self.backward[paf_i] = d7_b
            else:
                self.backward[paf_i] = np.concatenate((self.backward[paf_i], d7_b), axis=1)

            j2i = self.skeleton[paf_i][1] - 1
            if self.pif_floor < 1.0:
                pifhr_f = scalar_values(
                    self.pifhr[j2i], fourds[1, 1], fourds[1, 2], default=0.0)
                scores_f = scores * (self.pif_floor + (1.0 - self.pif_floor) * pifhr_f)
            else:
                scores_f = scores
            mask_f = scores_f > self.score_th
            d7_f = np.concatenate((
                np.expand_dims(scores_f[mask_f], 0),
                fourds[0, 1:4][:, mask_f],
                fourds[1, 1:4][:, mask_f],
            ))
            if self.forward[paf_i] is None:
                self.forward[paf_i] = d7_f
            else:
                self.forward[paf_i] = np.concatenate((self.forward[paf_i], d7_f), axis=1)

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
