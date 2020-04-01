import logging
import time

import numpy as np

LOG = logging.getLogger(__name__)


class CafSeeds:
    def __init__(self, seed_threshold, *, keypoints, skeleton,
                 score_scale=1.0,
                 debug_visualizer=None):
        self.seed_threshold = seed_threshold
        self.score_scale = score_scale
        self.keypoints = keypoints
        self.skeleton_m1 = np.array(skeleton) - 1
        self.debug_visualizer = debug_visualizer

        LOG.debug('seed threshold = %f', self.seed_threshold)

        self.seeds = []
        self.seed_values = []

    def fill(self, caf, stride=1.0):
        start = time.perf_counter()

        for field_i, p in enumerate(caf):
            p = p[:, :, p[0][0] > self.seed_threshold]
            (v1, x1, y1, _, s1), (__, x2, y2, ___, s2) = p

            j1i, j2i = self.skeleton_m1[field_i]
            new_seeds = np.zeros((len(v1), len(self.keypoints), 4), dtype=np.float32)
            for new_seed, vv, xx1, yy1, ss1, xx2, yy2, ss2 in zip(
                    new_seeds, v1, x1, y1, s1, x2, y2, s2):
                new_seed[j1i] = xx1, yy1, ss1, vv
                new_seed[j2i] = xx2, yy2, ss2, vv
                self.seed_values.append(vv)

            new_seeds[:, :, 0:3] *= stride
            self.seeds.append(new_seeds)

        LOG.debug('seeds %d, %.3fs', sum(len(s) for s in self.seeds), time.perf_counter() - start)
        return self

    def get(self):
        if self.debug_visualizer:
            self.debug_visualizer.seeds(self.seeds)

        order = np.argsort(self.seed_values)[::-1]
        return np.concatenate(self.seeds, axis=0)[order]

    def fill_sequence(self, cafs, strides=None):
        if strides is None:
            strides = [1.0 for _ in cafs]
        for caf, stride in zip(cafs, strides):
            self.fill(caf, stride)

        return self
