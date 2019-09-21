"""Decoder for pif-paf fields."""

from collections import defaultdict
import logging
from queue import PriorityQueue
import time

import numpy as np

from .annotation import Annotation
from .decoder import Decoder
from .paf_scored import PafScored
from .pif_hr import PifHr
from .pif_seeds import PifSeeds
from .utils import scalar_square_add_single, normalize_pif, normalize_paf

# pylint: disable=import-error
from ..functional import paf_center, scalar_value, scalar_nonzero, weiszfeld_nd

LOG = logging.getLogger(__name__)


class PifPaf4(Decoder):
    default_force_complete = True
    default_connection_method = 'max'
    default_fixed_b = None
    default_pif_fixed_scale = None
    default_paf_th = 0.1

    def __init__(self, stride, *,
                 seed_threshold=0.2,
                 head_indices=None,
                 skeleton=None,
                 confidence_scales=None,
                 debug_visualizer=None):
        self.head_indices = head_indices
        self.skeleton = skeleton

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer
        self.force_complete = self.default_force_complete
        self.connection_method = self.default_connection_method
        self.fixed_b = self.default_fixed_b
        self.pif_fixed_scale = self.default_pif_fixed_scale

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35
        self.paf_th = self.default_paf_th

        self.confidence_scales = confidence_scales

    # @classmethod
    # def cli(cls, parser):
    #     group = parser.add_argument_group('PifPaf decoder')
    #     group.add_argument('--fixed-b', default=None, type=float,
    #                        help='overwrite b with fixed value, e.g. 0.5')
    #     group.add_argument('--pif-fixed-scale', default=None, type=float,
    #                        help='overwrite pif scale with a fixed value')
    #     group.add_argument('--paf-th', default=cls.default_paf_th, type=float,
    #                        help='paf threshold')
    #     group.add_argument('--connection-method',
    #                        default=cls.default_connection_method,
    #                        choices=('median', 'max', 'blend'),
    #                        help='connection method to use, max is faster')

    @classmethod
    def apply_args(cls, args):
        cls.default_fixed_b = args.fixed_b
        cls.default_pif_fixed_scale = args.pif_fixed_scale
        cls.default_paf_th = args.paf_th
        cls.default_connection_method = args.connection_method

        # arg defined in factory
        cls.default_force_complete = args.force_complete_pose

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        pif, paf = fields[self.head_indices[0]], fields[self.head_indices[1]]
        if self.confidence_scales:
            cs = np.array(self.confidence_scales).reshape((-1, 1, 1))
            # print(paf[0].shape, cs.shape)
            # print('applying cs', cs)
            paf[0] = np.copy(paf[0])
            paf[0] *= cs
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
            self.debug_visualizer.paf_raw(paf, self.stride, reg_components=3)
        paf = normalize_paf(*paf, fixed_b=self.fixed_b)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)
        pifhr = PifHr(self.pif_nn).fill(pif, self.stride)
        seeds = PifSeeds(pifhr.targets, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer).fill(pif, self.stride).get()
        paf_scored = PafScored(pifhr.targets, self.skeleton, self.paf_th).fill(paf, self.stride)

        gen = PifPafGenerator(
            pifhr, paf_scored, seeds,
            stride=self.stride,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            paf_nn=self.paf_nn,
            paf_th=self.paf_th,
            skeleton=self.skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        if self.force_complete:
            paf_scored_c = PafScored(
                pifhr.targets, self.skeleton, score_th=0.0001).fill(paf, self.stride)
            gen.paf_scored = paf_scored_c
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations


class PifPafGenerator(object):
    def __init__(self, pifhr, paf_scored, seeds, *,
                 stride,
                 seed_threshold,
                 connection_method,
                 paf_nn,
                 paf_th,
                 skeleton,
                 debug_visualizer=None):
        self.paf_scored = paf_scored
        self.seeds = seeds

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.connection_method = connection_method
        self.paf_nn = paf_nn
        self.paf_th = paf_th
        self.skeleton = skeleton
        self.skeleton_m1 = np.asarray(skeleton) - 1

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # pif init
        self._pifhr, self._pifhr_scales = pifhr.clipped()
        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self._pifhr)

    def annotations(self, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        occupied = np.zeros(self._pifhr_scales.shape, dtype=np.uint8)
        annotations = []

        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                scalar_square_add_single(occupied[joint_i],
                                         xyv[0] * self.stride,
                                         xyv[1] * self.stride,
                                         max(4.0, width * self.stride),
                                         1)

        for ann in initial_annotations:
            if ann.joint_scales is None:
                ann.fill_joint_scales(self._pifhr_scales, self.stride)
            self._grow(ann, self.paf_th)
            ann.fill_joint_scales(self._pifhr_scales, self.stride)
            annotations.append(ann)
            mark_occupied(ann)

        for v, f, x, y, s in self.seeds:
            if scalar_nonzero(occupied[f], x, y):
                continue
            scalar_square_add_single(occupied[f], x, y, max(4.0, s), 1)

            ann = Annotation(self.skeleton).add(f, (x / self.stride, y / self.stride, v))
            self._grow(ann, self.paf_th)
            ann.fill_joint_scales(self._pifhr_scales, self.stride)
            annotations.append(ann)
            mark_occupied(ann)

        if self.debug_visualizer:
            LOG.debug('occupied field 0')
            self.debug_visualizer.occupied(occupied[0])

        LOG.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations

    def _grow_connection(self, xy, xy_scale, paf_field):
        assert len(xy) == 2
        assert paf_field.shape[0] == 7

        # source value
        paf_field = paf_center(paf_field, xy[0], xy[1], sigma=2.0 * xy_scale)
        if paf_field.shape[1] == 0:
            return 0, 0, 0

        # source distance
        d = np.linalg.norm(((xy[0],), (xy[1],)) - paf_field[1:3], axis=0)

        # combined value and source distance
        v = paf_field[0]
        scores = np.exp(-1.0 * d / xy_scale) * v  # two-tailed cumulative Laplace

        if self.connection_method == 'median':
            return self._target_with_median(paf_field[4:6], scores, sigma=1.0)
        if self.connection_method == 'max':
            return self._target_with_maxscore(paf_field[4:7], scores)
        if self.connection_method == 'blend':
            return self._target_with_blend(paf_field[4:7], scores)
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

    @staticmethod
    def _target_with_blend(target_coordinates, scores):
        """Blending the top two candidates with a weighted average.

        Similar to the post processing step in
        "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs".
        """
        assert target_coordinates.shape[1] == len(scores)
        if len(scores) == 1:
            return target_coordinates[0, 0], target_coordinates[1, 0], scores[0]

        sorted_i = np.argsort(scores)
        max_entry_1 = target_coordinates[:, sorted_i[-1]]
        max_entry_2 = target_coordinates[:, sorted_i[-2]]

        score_1 = scores[sorted_i[-1]]
        score_2 = scores[sorted_i[-2]]
        return (
            (score_1 * max_entry_1[0] + score_2 * max_entry_2[0]) / (score_1 + score_2),
            (score_1 * max_entry_1[1] + score_2 * max_entry_2[1]) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        )

    def connection_value(self, ann, paf_i, forward, th, reverse_match=True):
        j1i, j2i = self.skeleton_m1[paf_i]
        if forward:
            jsi, jti = j1i, j2i
            directed_paf_field = self.paf_scored.forward[paf_i]
            directed_paf_field_reverse = self.paf_scored.backward[paf_i]
        else:
            jsi, jti = j2i, j1i
            directed_paf_field = self.paf_scored.backward[paf_i]
            directed_paf_field_reverse = self.paf_scored.forward[paf_i]
        xyv = ann.data[jsi]
        xy_scale_s = max(
            1.0,
            scalar_value(self._pifhr_scales[jsi],
                         xyv[0] * self.stride,
                         xyv[1] * self.stride) / self.stride
        )

        new_xyv = self._grow_connection(xyv[:2], xy_scale_s, directed_paf_field)
        if new_xyv[2] < th:
            return 0.0, 0.0, 0.0
        xy_scale_t = max(
            1.0,
            scalar_value(self._pifhr_scales[jti],
                         new_xyv[0] * self.stride,
                         new_xyv[1] * self.stride) / self.stride
        )

        # reverse match
        if reverse_match:
            reverse_xyv = self._grow_connection(
                new_xyv[:2], xy_scale_t, directed_paf_field_reverse)
            if reverse_xyv[2] < th:
                return 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0

        return (new_xyv[0], new_xyv[1], np.sqrt(new_xyv[2] * xyv[2]))  # geometric mean

    def _grow(self, ann, th, reverse_match=True):
        LOG.debug('_____________new grow_________')
        frontier = PriorityQueue()
        evaluated_connections = set()

        LOG.debug('skeleton connections = %d', len(self.skeleton_m1))

        def add_to_frontier(start_i):
            for paf_i, (j1, j2) in enumerate(self.skeleton_m1):
                if j1 == start_i:
                    end_i = j2
                    forward = True
                elif j2 == start_i:
                    end_i = j1
                    forward = False
                else:
                    continue

                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in evaluated_connections:
                    continue

                LOG.debug('computing value for %d to %d', start_i, end_i)
                new_xyv = self.connection_value(
                    ann, paf_i, forward, th, reverse_match=reverse_match)
                if new_xyv[2] == 0.0:
                    continue
                frontier.put((-new_xyv[2], new_xyv, start_i, end_i))
                evaluated_connections.add((start_i, end_i))

        # seeding the frontier
        for joint_i, v in enumerate(ann.data[:, 2]):
            if v == 0.0:
                continue
            add_to_frontier(joint_i)

        while frontier.qsize():
            _, new_xyv, jsi, jti = frontier.get()
            if ann.data[jti, 2] > 0.0:
                continue

            LOG.debug('setting joint %d: %s', jti, new_xyv)

            ann.data[jti] = new_xyv
            ann.decoding_order.append(
                (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
            add_to_frontier(jti)

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

        for ann in annotations:
            unfilled_mask = ann.data[:, 2] == 0.0
            self._grow(ann, th=1e-8, reverse_match=False)
            now_filled_mask = ann.data[:, 2] > 0.0
            updated = np.logical_and(unfilled_mask, now_filled_mask)
            ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])
            ann.fill_joint_scales(self._pifhr_scales, self.stride)

            # some joints might still be unfilled
            if np.any(ann.data[:, 2] == 0.0):
                self._flood_fill(ann)

        LOG.debug('complete annotations %.3fs', time.perf_counter() - start)
        return annotations
