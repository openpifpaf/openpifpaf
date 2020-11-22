import logging
import time
from typing import List

import numpy as np

# pylint: disable=import-error
from ...functional import scalar_values
from ... import headmeta

LOG = logging.getLogger(__name__)


class CafScored:
    default_score_th = 0.1
    ablation_no_rescore = False

    def __init__(self, cifhr, *, score_th=None, cif_floor=0.1):
        self.cifhr = cifhr
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        self.forward = None
        self.backward = None

    def directed(self, caf_i, forward):
        if forward:
            return self.forward[caf_i], self.backward[caf_i]

        return self.backward[caf_i], self.forward[caf_i]

    def rescore(self, nine, joint_t):
        if self.cif_floor < 1.0 and joint_t < len(self.cifhr) and not self.ablation_no_rescore:
            cifhr_t = scalar_values(self.cifhr[joint_t], nine[3], nine[4], default=0.0)
            nine[0] = nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_t)
        return nine[:, nine[0] > self.score_th]

    def fill_single(self, all_fields, meta: headmeta.Caf):
        start = time.perf_counter()
        caf = all_fields[meta.head_index]

        if self.forward is None:
            self.forward = [np.empty((9, 0), dtype=caf.dtype) for _ in caf]
            self.backward = [np.empty((9, 0), dtype=caf.dtype) for _ in caf]

        for caf_i, nine in enumerate(caf):
            assert nine.shape[0] == 9

            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]

            if meta.decoder_min_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist > meta.decoder_min_distance / meta.stride
                nine = nine[:, mask_dist]

            if meta.decoder_max_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist < meta.decoder_max_distance / meta.stride
                nine = nine[:, mask_dist]

            nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= meta.stride

            nine_b = np.copy(nine[(0, 3, 4, 1, 2, 6, 5, 8, 7), :])
            nine_b = self.rescore(nine_b, meta.skeleton[caf_i][0] - 1)
            self.backward[caf_i] = np.concatenate((self.backward[caf_i], nine_b), axis=1)

            nine_f = np.copy(nine)
            nine_f = self.rescore(nine_f, meta.skeleton[caf_i][1] - 1)
            self.forward[caf_i] = np.concatenate((self.forward[caf_i], nine_f), axis=1)

        LOG.debug('scored caf (%d, %d) in %.3fs',
                  sum(f.shape[1] for f in self.forward),
                  sum(b.shape[1] for b in self.backward),
                  time.perf_counter() - start)
        return self

    def fill(self, all_fields, metas: List[headmeta.Caf]):
        for meta in metas:
            self.fill_single(all_fields, meta)

        return self
