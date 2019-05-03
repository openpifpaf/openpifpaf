"""Decoder for pif fields."""

from collections import defaultdict
import logging
import time

import numpy as np

from .annotation import AnnotationWithoutSkeleton
from .decoder import Decoder
from .utils import index_field, scalar_square_add_single, normalize_pif

# pylint: disable=import-error
from ..functional import (scalar_square_add_constant, scalar_square_add_gauss)


class Pif(Decoder):
    default_pif_fixed_scale = None

    def __init__(self, stride, seed_threshold,
                 head_index=None,
                 profile=None,
                 debug_visualizer=None,
                 **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments %s', kwargs)

        self.stride = stride
        self.hr_scale = self.stride
        self.head_index = head_index or 0
        self.profile = profile
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn = 16

    @staticmethod
    def match(head_names):
        return head_names in (
            ('pif',),
        )

    @classmethod
    def apply_args(cls, args):
        cls.default_pif_fixed_scale = args.pif_fixed_scale

    def __call__(self, fields):
        start = time.perf_counter()
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

        self.log.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        if self.profile is not None:
            self.profile.disable()
        return annotations


class PifGenerator(object):
    def __init__(self, pif_field, *,
                 stride,
                 seed_threshold,
                 pif_nn,
                 debug_visualizer=None):
        self.log = logging.getLogger(self.__class__.__name__)

        self.pif = pif_field

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.pif_nn = pif_nn
        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # pif init
        self._pifhr, self._pifhr_scales = self._target_intensities()
        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self._pifhr)

    def _target_intensities(self, v_th=0.1):
        start = time.perf_counter()

        targets = np.zeros((self.pif.shape[0],
                            int(self.pif.shape[2] * self.stride),
                            int(self.pif.shape[3] * self.stride)))
        scales = np.zeros(targets.shape)
        ns = np.zeros(targets.shape)
        for t, p, scale, n in zip(targets, self.pif, scales, ns):
            v, x, y, s = p[:, p[0] > v_th]
            x = x * self.stride
            y = y * self.stride
            s = s * self.stride
            scalar_square_add_gauss(t, x, y, s, v / self.pif_nn, truncate=0.5)
            scalar_square_add_constant(scale, x, y, s, s*v)
            scalar_square_add_constant(n, x, y, s, v)

        targets = np.minimum(1.0, targets)

        m = ns > 0
        scales[m] = scales[m] / ns[m]
        self.log.debug('target_intensities %.3fs', time.perf_counter() - start)
        return targets, scales

    def annotations(self):
        start = time.perf_counter()

        seeds = self._pifhr_seeds()
        annotations = []
        for v, f, x, y in seeds:
            ann = AnnotationWithoutSkeleton(f, (x, y, v), self._pifhr_scales.shape[0])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)
            annotations.append(ann)

        self.log.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations

    def _pifhr_seeds(self):
        start = time.perf_counter()
        seeds = []
        for field_i, (f, s) in enumerate(zip(self._pifhr, self._pifhr_scales)):
            index_fields = index_field(f.shape)
            candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)

            mask = f > self.seed_threshold
            candidates = np.moveaxis(candidates[:, mask], 0, -1)

            occupied = np.zeros(s.shape)
            for c in sorted(candidates, key=lambda c: c[2], reverse=True):
                i, j = int(c[0]), int(c[1])
                if occupied[j, i]:
                    continue

                width = max(4, s[j, i])
                scalar_square_add_single(occupied, c[0], c[1], width / 2.0, 1.0)
                seeds.append((c[2], field_i, c[0] / self.stride, c[1] / self.stride))

            if self.debug_visualizer:
                if field_i in self.debug_visualizer.pif_indices:
                    self.log.debug('occupied seed, field %d', field_i)
                    self.debug_visualizer.occupied(occupied)

        seeds = list(sorted(seeds, reverse=True))
        if len(seeds) > 500:
            if seeds[500][0] > 0.1:
                seeds = [s for s in seeds if s[0] > 0.1]
            else:
                seeds = seeds[:500]

        if self.debug_visualizer:
            self.debug_visualizer.seeds(seeds, self.stride)

        self.log.debug('seeds %d, %.3fs', len(seeds), time.perf_counter() - start)
        return seeds
