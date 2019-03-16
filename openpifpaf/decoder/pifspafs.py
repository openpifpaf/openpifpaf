"""Decoder for pifs-pafs fields."""

from collections import defaultdict
import time

import numpy as np

from .annotation import Annotation
from .utils import (index_field, scalar_square_add_single,
                    normalize_pifs, normalize_paf)
from ..data import COCO_PERSON_SKELETON

# pylint: disable=import-error
from ..functional import (scalar_square_add_constant, scalar_square_add_gauss,
                          weiszfeld_nd, paf_mask_center)


class PifsPafs(object):
    def __init__(self, stride, seed_threshold,
                 skeleton=None, head_indices=None,
                 profile=None,
                 force_complete=True,
                 debug_visualizer=None,
                 connection_method='max',
                 pif_fixed_scale=None):
        self.stride = stride
        self.hr_scale = self.stride
        self.skeleton = skeleton or COCO_PERSON_SKELETON
        self.head_indices = head_indices
        self.profile = profile
        self.seed_threshold = seed_threshold
        self.force_complete = force_complete
        self.debug_visualizer = debug_visualizer
        self.connection_method = connection_method
        self.pif_fixed_scale = pif_fixed_scale

        self.pif_nn = 16
        self.paf_nn = 1 if connection_method == 'max' else 35

    def __call__(self, fields):
        start = time.time()
        if self.profile is not None:
            self.profile.enable()

        if not self.head_indices:
            pif, paf = fields
        else:
            pif, paf = fields[self.head_indices[0]], fields[self.head_indices[1]]
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
            self.debug_visualizer.paf_raw(paf, self.stride, reg_components=3)
        paf = normalize_paf(*paf)
        pif = normalize_pifs(*pif, fixed_scale=self.pif_fixed_scale)

        gen = PifsPafsGenerator(
            pif, paf,
            stride=self.stride,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            pif_nn=self.pif_nn,
            paf_nn=self.paf_nn,
            skeleton=self.skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations()
        if self.force_complete:
            annotations = gen.complete_annotations(annotations)

        print('annotations', len(annotations), time.time() - start)
        if self.profile is not None:
            self.profile.disable()
        return annotations


class PifsPafsGenerator(object):
    def __init__(self, pifs_field, pafs_field, *,
                 stride,
                 seed_threshold,
                 connection_method,
                 pif_nn,
                 paf_nn,
                 skeleton,
                 debug_visualizer=None):
        self.pif = pifs_field
        self.paf = pafs_field

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.connection_method = connection_method
        self.pif_nn = pif_nn
        self.paf_nn = paf_nn
        self.skeleton = skeleton

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

        # paf init
        self._paf_forward = None
        self._paf_backward = None
        self._paf_forward, self._paf_backward = self._score_paf_target()


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

    def _score_paf_target(self, pifhr_floor=0.01, score_th=0.1):
        start = time.time()

        scored_forward = []
        scored_backward = []
        for c, fourds in enumerate(self.paf):
            assert fourds.shape[0] == 2
            assert fourds.shape[1] == 4

            scores = np.min(fourds[:, 0], axis=0)
            mask = scores > score_th
            scores = scores[mask]
            fourds = fourds[:, :, mask]

            j1i = self.skeleton[c][0] - 1
            if pifhr_floor < 1.0:
                ij_b = np.round(fourds[0, 1:3] * self.stride).astype(np.int)
                ij_b[0] = np.clip(ij_b[0], 0, self._pifhr.shape[2] - 1)
                ij_b[1] = np.clip(ij_b[1], 0, self._pifhr.shape[1] - 1)
                pifhr_b = self._pifhr[j1i, ij_b[1], ij_b[0]]
                scores_b = scores * (pifhr_floor + (1.0 - pifhr_floor) * pifhr_b)
            else:
                scores_b = scores
            mask_b = scores_b > score_th
            scored_backward.append(np.concatenate((
                np.expand_dims(scores_b[mask_b], 0),
                fourds[1, 1:4][:, mask_b],
                fourds[0, 1:4][:, mask_b],
            )))

            j2i = self.skeleton[c][1] - 1
            if pifhr_floor < 1.0:
                ij_f = np.round(fourds[1, 1:3] * self.stride).astype(np.int)
                ij_f[0] = np.clip(ij_f[0], 0, self._pifhr.shape[2] - 1)
                ij_f[1] = np.clip(ij_f[1], 0, self._pifhr.shape[1] - 1)
                pifhr_f = self._pifhr[j2i, ij_f[1], ij_f[0]]
                scores_f = scores * (pifhr_floor + (1.0 - pifhr_floor) * pifhr_f)
            else:
                scores_f = scores
            mask_f = scores_f > score_th
            scored_forward.append(np.concatenate((
                np.expand_dims(scores_f[mask_f], 0),
                fourds[0, 1:4][:, mask_f],
                fourds[1, 1:4][:, mask_f],
            )))

        print('scored paf', time.time() - start)
        return scored_forward, scored_backward

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

            ann = Annotation(f, (x, y, v), self.skeleton)
            self._grow(ann, self._paf_forward, self._paf_backward)
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

    def _grow_connection(self, xy, paf_field):
        assert len(xy) == 2
        assert paf_field.shape[0] == 7

        # source value
        s_mask = paf_mask_center(paf_field, xy[0], xy[1], sigma=2.0)
        if not np.any(s_mask):
            return 0, 0, 0
        paf_field = paf_field[:, s_mask]

        # source distance
        d = np.linalg.norm(np.expand_dims(xy, 1) - paf_field[1:3], axis=0)
        b_source = paf_field[3]
        b_target = paf_field[6]

        # combined value and source distance
        v = paf_field[0]
        scores = np.exp(-1.0 * d / b_source) * v  # two-tailed cumulative Laplace

        if self.connection_method == 'median':
            return self._target_with_median(paf_field[4:6], scores, sigma=1.0)
        if self.connection_method == 'max':
            return self._target_with_maxscore(paf_field[4:7], scores)
        raise Exception('connection method not known')

    def _target_with_median(self, target_coordinates, scores, sigma, max_steps=20):
        target_coordinates = np.moveaxis(target_coordinates, 0, -1)
        assert target_coordinates.shape[0] == scores.shape[0]

        if target_coordinates.shape[0] == 1:
            return (target_coordinates[0][0],
                    target_coordinates[0][1],
                    np.tanh(scores[0] * 3.0 / self.paf_nn))

        y = np.sum(target_coordinates * np.expand_dims(scores, -1), axis=0) / np.sum(scores)
        if target_coordinates.shape[0] == 2:
            return y[0], y[1], np.tanh(np.sum(scores) * 3.0 / self.paf_nn)
        y, prev_d = weiszfeld_nd(target_coordinates, y, weights=scores, max_steps=max_steps)

        closest = prev_d < sigma
        close_scores = np.sort(scores[closest])[-self.paf_nn:]
        score = np.tanh(np.sum(close_scores) * 3.0 / self.paf_nn)
        return (y[0], y[1], score)

    @staticmethod
    def _target_with_maxscore(target_coordinates, scores):
        assert target_coordinates.shape[1] == scores.shape[0]

        max_i = np.argmax(scores)
        max_entry = target_coordinates[:, max_i]

        score = scores[max_i]
        return max_entry[0], max_entry[1], score

    def _grow(self, ann, paf_forward, paf_backward, th=0.1):
        for _, i, forward, j1i, j2i in ann.frontier_iter():
            if forward:
                xyv = ann.data[j1i]
                directed_paf_field = paf_forward[i]
                directed_paf_field_reverse = paf_backward[i]
            else:
                xyv = ann.data[j2i]
                directed_paf_field = paf_backward[i]
                directed_paf_field_reverse = paf_forward[i]

            new_xyv = self._grow_connection(xyv[:2], directed_paf_field)
            if new_xyv[2] < th:
                continue

            # reverse match
            if th >= 0.1:
                reverse_xyv = self._grow_connection(new_xyv[:2], directed_paf_field_reverse)
                if reverse_xyv[2] < th:
                    continue
                if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > 1.0:
                    continue

            new_xyv = (new_xyv[0], new_xyv[1], np.sqrt(new_xyv[2] * xyv[2]))  # geometric mean
            if forward:
                if new_xyv[2] > ann.data[j2i, 2]:
                    ann.data[j2i] = new_xyv
            else:
                if new_xyv[2] > ann.data[j1i, 2]:
                    ann.data[j1i] = new_xyv

    def complete_annotations(self, annotations):
        start = time.time()

        paf_forward_c, paf_backward_c = self._score_paf_target(
            pifhr_floor=0.9, score_th=0.0001)

        for ann in annotations:
            unfilled_mask = ann.data[:, 2] == 0.0
            self._grow(ann, paf_forward_c, paf_backward_c, th=1e-8)
            now_filled_mask = ann.data[:, 2] > 0.0
            updated = np.logical_and(unfilled_mask, now_filled_mask)
            ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)

        print('complete annotations', time.time() - start)
        return annotations
