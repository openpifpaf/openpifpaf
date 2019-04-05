"""Decoder for pif fields."""

from collections import defaultdict
import time

import numpy as np

from .annotation import AnnotationWithoutSkeleton
from .utils import index_field, scalar_square_add_single, normalize_pif

# pylint: disable=import-error
from ..functional import (scalar_square_add_constant, scalar_square_add_gauss)


class Pif(object):
    def __init__(self, stride, seed_threshold,
                 head_index=None,
                 profile=None,
                 debug_visualizer=None,
                 pif_fixed_scale=None):
        self.stride = stride
        self.hr_scale = self.stride
        self.head_index = head_index or 0
        self.profile = profile
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.pif_fixed_scale = pif_fixed_scale

        self.pif_nn = 16

    def __call__(self, fields):
        start = time.time()
        if self.profile is not None:
            self.profile.enable()

        pif = fields[self.head_index]
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)

        gen = PifGenerator(
            pif,
            stride=self.stride,
            seed_threshold=self.seed_threshold,
            pif_nn=self.pif_nn,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations()

        print('annotations', len(annotations), time.time() - start)
        if self.profile is not None:
            self.profile.disable()
        return annotations


class PifGenerator(object):
    def __init__(self, pif_field, *,
                 stride,
                 seed_threshold,
                 pif_nn,
                 debug_visualizer=None):
        self.pif = pif_field

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.pif_nn = pif_nn
        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # pif init
        self._pifhr = None
        self._pifhr_scales = None
        self._pifhr_core = None
        self._pifhr, self._pifhr_scales = self._target_intensities()
        self._pifhr_core = self._target_intensities(core_only=True)
        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self._pifhr)
            self.debug_visualizer.pifhr(self._pifhr_core)

    def _target_intensities(self, v_th=0.1, core_only=False):
        start = time.time()

        targets = np.zeros((self.pif.shape[0],
                            int(self.pif.shape[2] * self.stride),
                            int(self.pif.shape[3] * self.stride)))
        scales = np.zeros_like(targets)
        ns = np.zeros_like(targets)
        for t, p, scale, n in zip(targets, self.pif, scales, ns):
            v, x, y, s = p[:, p[0] > v_th]
            x = x * self.stride
            y = y * self.stride
            s = s * self.stride
            if core_only:
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn, truncate=0.5)
            else:
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn)
                scalar_square_add_constant(scale, x, y, s, s*v)
                scalar_square_add_constant(n, x, y, s, v)

        targets = np.minimum(1.0, targets)
        if core_only:
            print('target_intensities', time.time() - start)
            return targets

        m = ns > 0
        scales[m] = scales[m] / ns[m]
        print('target_intensities', time.time() - start)
        return targets, scales

    def annotations(self):
        start = time.time()

        seeds = self._pifhr_seeds()

        occupied = np.zeros_like(self._pifhr_scales)
        annotations = []
        for v, f, x, y in seeds:
            i = np.clip(int(round(x * self.stride)), 0, occupied.shape[2] - 1)
            j = np.clip(int(round(y * self.stride)), 0, occupied.shape[1] - 1)
            if occupied[f, j, i]:
                continue

            ann = AnnotationWithoutSkeleton(f, (x, y, v), self._pifhr_scales.shape[0])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)
            annotations.append(ann)

            for i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[i] * self.stride
                scalar_square_add_single(occupied[i],
                                         xyv[0] * self.stride,
                                         xyv[1] * self.stride,
                                         width / 2.0,
                                         1.0)

        if self.debug_visualizer:
            print('occupied annotations field 0')
            self.debug_visualizer.occupied(occupied[0])

        print('keypoint sets', len(annotations), time.time() - start)
        return annotations

    def _pifhr_seeds(self):
        start = time.time()
        seeds = []
        for field_i, (f, s) in enumerate(zip(self._pifhr_core, self._pifhr_scales)):
            index_fields = index_field(f.shape)
            candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)

            mask = f > self.seed_threshold
            candidates = np.moveaxis(candidates[:, mask], 0, -1)

            occupied = np.zeros_like(s)
            for c in sorted(candidates, key=lambda c: c[2], reverse=True):
                i, j = int(c[0]), int(c[1])
                if occupied[j, i]:
                    continue

                width = max(4, s[j, i])
                scalar_square_add_single(occupied, c[0], c[1], width / 2.0, 1.0)
                seeds.append((c[2], field_i, c[0] / self.stride, c[1] / self.stride))

            if self.debug_visualizer:
                if field_i in self.debug_visualizer.pif_indices:
                    print('occupied seed, field {}'.format(field_i))
                    self.debug_visualizer.occupied(occupied)

        seeds = list(sorted(seeds, reverse=True))
        if len(seeds) > 500:
            if seeds[500][0] > 0.1:
                seeds = [s for s in seeds if s[0] > 0.1]
            else:
                seeds = seeds[:500]

        if self.debug_visualizer:
            self.debug_visualizer.seeds(seeds, self.stride)

        print('seeds', len(seeds), time.time() - start)
        return seeds
