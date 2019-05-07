"""Decoder for pif-paf fields."""

from collections import defaultdict
import logging
import time

import numpy as np

from .annotation import Annotation
from .decoder import Decoder
from .utils import (index_field, scalar_square_add_single,
                    normalize_pif, normalize_paf)
from ..data import KINEMATIC_TREE_SKELETON, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_SKELETON

# pylint: disable=import-error
from ..functional import (scalar_square_add_constant, scalar_square_add_gauss,
                          weiszfeld_nd, paf_center)


class PifPaf(Decoder):
    default_force_complete = True
    default_connection_method = 'max'
    default_fixed_b = None
    default_pif_fixed_scale = None

    def __init__(self, stride, *,
                 seed_threshold=0.2,
                 head_names=None,
                 head_indices=None,
                 skeleton=None,
                 debug_visualizer=None,
                 **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('unused arguments %s', kwargs)

        if head_names is None:
            head_names = ('pif', 'paf')

        self.head_indices = head_indices
        if self.head_indices is None:
            self.head_indices = {
                ('paf', 'pif', 'paf'): [1, 2],
                ('pif', 'pif', 'paf'): [1, 2],
            }.get(head_names, [0, 1])

        self.skeleton = skeleton
        if self.skeleton is None:
            paf_name = head_names[self.head_indices[1]]
            if paf_name == 'paf16':
                self.skeleton = KINEMATIC_TREE_SKELETON
            elif paf_name == 'paf44':
                self.skeleton = DENSER_COCO_PERSON_SKELETON
            else:
                self.skeleton = COCO_PERSON_SKELETON

        self.stride = stride
        self.hr_scale = self.stride
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.force_complete = self.default_force_complete
        self.connection_method = self.default_connection_method
        self.fixed_b = self.default_fixed_b
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

    @staticmethod
    def match(head_names):
        return head_names in (
            ('pif', 'paf'),
            ('pif', 'paf44'),
            ('pif', 'paf16'),
            ('paf', 'pif', 'paf'),
            ('pif', 'pif', 'paf'),
            ('pif', 'wpaf'),
        )

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('PifPaf decoder')
        group.add_argument('--fixed-b', default=None, type=float,
                           help='overwrite b with fixed value, e.g. 0.5')
        group.add_argument('--pif-fixed-scale', default=None, type=float,
                           help='overwrite pif scale with a fixed value')
        group.add_argument('--connection-method',
                           default='max', choices=('median', 'max'),
                           help='connection method to use, max is faster')

    @classmethod
    def apply_args(cls, args):
        cls.default_fixed_b = args.fixed_b
        cls.default_pif_fixed_scale = args.pif_fixed_scale
        cls.default_connection_method = args.connection_method

        # arg defined in factory
        cls.default_force_complete = args.force_complete_pose

    def __call__(self, fields):
        start = time.perf_counter()

        pif, paf = fields[self.head_indices[0]], fields[self.head_indices[1]]
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
            self.debug_visualizer.paf_raw(paf, self.stride, reg_components=3)
        paf = normalize_paf(*paf, fixed_b=self.fixed_b)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)

        gen = PifPafGenerator(
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

        self.log.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations


class PifPafGenerator(object):
    def __init__(self, pifs_field, pafs_field, *,
                 stride,
                 seed_threshold,
                 connection_method,
                 pif_nn,
                 paf_nn,
                 skeleton,
                 debug_visualizer=None):
        self.log = logging.getLogger(self.__class__.__name__)

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
            if core_only:
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn, truncate=0.5)
            else:
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn)
                scalar_square_add_constant(scale, x, y, s, s*v)
                scalar_square_add_constant(n, x, y, s, v)

        if core_only:
            self.log.debug('target_intensities %.3fs', time.perf_counter() - start)
            return targets

        m = ns > 0
        scales[m] = scales[m] / ns[m]
        self.log.debug('target_intensities %.3fs', time.perf_counter() - start)
        return targets, scales

    def _score_paf_target(self, pifhr_floor=0.1, score_th=0.1):
        start = time.perf_counter()

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

        self.log.debug('scored paf %.3fs', time.perf_counter() - start)
        return scored_forward, scored_backward

    def annotations(self):
        start = time.perf_counter()

        seeds = self._pifhr_seeds()

        occupied = np.zeros(self._pifhr_scales.shape)
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
                                         max(4.0, width / 2.0),
                                         1.0)

        if self.debug_visualizer:
            self.log.debug('occupied annotations field 0')
            self.debug_visualizer.occupied(occupied[0])

        self.log.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations

    def _pifhr_seeds(self):
        start = time.perf_counter()
        seeds = []
        for field_i, (f, s) in enumerate(zip(self._pifhr_core, self._pifhr_scales)):
            index_fields = index_field(f.shape)

            f = f.reshape(-1)
            mask = f > self.seed_threshold
            f = f[mask]
            index_fields = index_fields.reshape(2, -1)[:, mask]

            occupied = np.zeros(s.shape)
            for entry_i in np.argsort(f)[::-1]:
                x, y = index_fields[:, entry_i]
                v = f[entry_i]

                i, j = int(x), int(y)
                if occupied[j, i]:
                    continue

                width = max(4, s[j, i])
                scalar_square_add_single(occupied, x, y, width / 2.0, 1.0)
                seeds.append((v, field_i, x / self.stride, y / self.stride))

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

    def _grow_connection(self, xy, paf_field):
        assert len(xy) == 2
        assert paf_field.shape[0] == 7

        # source value
        paf_field = paf_center(paf_field, xy[0], xy[1], sigma=2.0)
        if paf_field.shape[1] == 0:
            return 0, 0, 0

        # source distance
        d = np.linalg.norm(np.expand_dims(xy, 1) - paf_field[1:3], axis=0)
        b_source = paf_field[3] * 3.0
        # b_target = paf_field[6]

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

    @staticmethod
    def _flood_fill(ann):
        for _, _, forward, j1i, j2i in ann.frontier_iter():
            if forward:
                xyv_s = ann.data[j1i]
                xyv_t = ann.data[j2i]
            else:
                xyv_s = ann.data[j2i]
                xyv_t = ann.data[j1i]

            xyv_t[:2] = xyv_s[:2]
            xyv_t[2] = 0.00001

    def complete_annotations(self, annotations):
        start = time.perf_counter()

        paf_forward_c, paf_backward_c = self._score_paf_target(score_th=0.0001)
        for ann in annotations:
            unfilled_mask = ann.data[:, 2] == 0.0
            self._grow(ann, paf_forward_c, paf_backward_c, th=1e-8)
            now_filled_mask = ann.data[:, 2] > 0.0
            updated = np.logical_and(unfilled_mask, now_filled_mask)
            ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)

            # some joints might still be unfilled
            if np.any(ann.data[:, 2] == 0.0):
                self._flood_fill(ann)

        self.log.debug('complete annotations %.3fs', time.perf_counter() - start)
        return annotations
