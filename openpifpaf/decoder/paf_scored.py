import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import scalar_values

LOG = logging.getLogger(__name__)


class PafScored(object):
    def __init__(self, pifhr, skeleton, score_th, pif_floor=0.1):
        self.pifhr = pifhr
        self.skeleton = skeleton
        self.score_th = score_th
        self.pif_floor = pif_floor

        self.forward = []
        self.backward = []

    def fill(self, paf, stride):
        start = time.perf_counter()

        for c, fourds in enumerate(paf):
            assert fourds.shape[0] == 2
            assert fourds.shape[1] == 4

            scores = np.min(fourds[:, 0], axis=0)
            mask = scores > self.score_th
            scores = scores[mask]
            fourds = np.copy(fourds[:, :, mask])
            fourds[:, 1] *= stride
            fourds[:, 2] *= stride
            fourds[:, 3] *= stride

            j1i = self.skeleton[c][0] - 1
            if self.pif_floor < 1.0:
                pifhr_b = scalar_values(self.pifhr[j1i],
                                        fourds[0, 1],
                                        fourds[0, 2],
                                        default=0.0)
                scores_b = scores * (self.pif_floor + (1.0 - self.pif_floor) * pifhr_b)
            else:
                scores_b = scores
            mask_b = scores_b > self.score_th
            self.backward.append(np.concatenate((
                np.expand_dims(scores_b[mask_b], 0),
                fourds[1, 1:4][:, mask_b],
                fourds[0, 1:4][:, mask_b],
            )))

            j2i = self.skeleton[c][1] - 1
            if self.pif_floor < 1.0:
                pifhr_f = scalar_values(self.pifhr[j2i],
                                        fourds[1, 1],
                                        fourds[1, 2],
                                        default=0.0)
                scores_f = scores * (self.pif_floor + (1.0 - self.pif_floor) * pifhr_f)
            else:
                scores_f = scores
            mask_f = scores_f > self.score_th
            self.forward.append(np.concatenate((
                np.expand_dims(scores_f[mask_f], 0),
                fourds[0, 1:4][:, mask_f],
                fourds[1, 1:4][:, mask_f],
            )))

        LOG.debug('scored paf %.3fs', time.perf_counter() - start)
        return self
