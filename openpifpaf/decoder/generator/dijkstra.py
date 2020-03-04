from collections import defaultdict
import logging
from queue import PriorityQueue
import time

import numpy as np

from ..annotation import Annotation
from ..occupancy import Occupancy
from ..utils import scalar_square_add_single

# pylint: disable=import-error
from ...functional import paf_center_s, scalar_nonzero_clipped

LOG = logging.getLogger(__name__)


class Dijkstra(object):
    keypoint_threshold = 0.0

    def __init__(self, pifhr, paf_scored, seeds, *,
                 seed_threshold,
                 connection_method,
                 paf_nn,
                 paf_th,
                 keypoints,
                 skeleton,
                 out_skeleton=None,
                 confirm_connections=False,
                 confidence_scales=None,
                 debug_visualizer=None):
        self.pifhr = pifhr
        self.paf_scored = paf_scored
        self.seeds = seeds

        self.seed_threshold = seed_threshold
        self.connection_method = connection_method
        self.paf_nn = paf_nn
        self.paf_th = paf_th

        self.keypoints = keypoints
        self.skeleton = skeleton
        self.skeleton_m1 = np.asarray(skeleton) - 1
        self.out_skeleton = out_skeleton or skeleton
        self.confirm_connections = confirm_connections
        self.confidence_scales = confidence_scales

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(list)
        for paf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_target[j2].append((paf_i, True, j1))
            self.by_target[j1].append((paf_i, False, j2))
        self.by_source = defaultdict(list)
        for paf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_source[j1].append((paf_i, True, j2))
            self.by_source[j2].append((paf_i, False, j1))

        # pif init
        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self.pifhr.targets)

    def annotations(self, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        occupied = Occupancy(self.pifhr.targets.shape, 2, min_scale=4)
        annotations = []

        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)

        for ann in initial_annotations:
            self._grow(ann, self.paf_th)
            annotations.append(ann)
            mark_occupied(ann)

        for v, f, x, y, s in self.seeds.get():
            if occupied.get(f, x, y):
                continue

            ann = Annotation(self.keypoints, self.out_skeleton).add(f, (x, y, v))
            ann.joint_scales[f] = s
            self._grow(ann, self.paf_th)
            annotations.append(ann)
            mark_occupied(ann)

        if self.debug_visualizer:
            self.debug_visualizer.occupied(occupied)

        LOG.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations

    def _grow_connection(self, xy, xy_scale, paf_field):
        assert len(xy) == 2
        assert paf_field.shape[0] == 9

        # source value
        paf_field = paf_center_s(paf_field, xy[0], xy[1], sigma=2.0 * xy_scale)
        if paf_field.shape[1] == 0:
            return 0, 0, 0, 0

        # source distance
        d = np.linalg.norm(((xy[0],), (xy[1],)) - paf_field[1:3], axis=0)

        # combined value and source distance
        v = paf_field[0]
        scores = np.exp(-1.0 * d / xy_scale) * v  # two-tailed cumulative Laplace

        if self.connection_method == 'max':
            return self._target_with_maxscore(paf_field[5:], scores)
        if self.connection_method == 'blend':
            return self._target_with_blend(paf_field[5:], scores)
        raise Exception('connection method not known')

    @staticmethod
    def _target_with_maxscore(target_coordinates, scores):
        assert target_coordinates.shape[1] == scores.shape[0]

        max_i = np.argmax(scores)
        max_entry = target_coordinates[:, max_i]

        score = scores[max_i]
        return max_entry[0], max_entry[1], max_entry[3], score

    @staticmethod
    def _target_with_blend(target_coordinates, scores):
        """Blending the top two candidates with a weighted average.

        Similar to the post processing step in
        "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs".
        """
        assert target_coordinates.shape[1] == len(scores)
        if len(scores) == 1:
            return (
                target_coordinates[0, 0],
                target_coordinates[1, 0],
                target_coordinates[3, 0],
                scores[0],
            )

        sorted_i = np.argsort(scores)
        max_entry_1 = target_coordinates[:, sorted_i[-1]]
        max_entry_2 = target_coordinates[:, sorted_i[-2]]

        score_1 = scores[sorted_i[-1]]
        score_2 = scores[sorted_i[-2]]
        if score_2 < 0.01 or score_2 < 0.5 * score_1:
            return max_entry_1[0], max_entry_1[1], max_entry_1[3], score_1

        return (
            (score_1 * max_entry_1[0] + score_2 * max_entry_2[0]) / (score_1 + score_2),
            (score_1 * max_entry_1[1] + score_2 * max_entry_2[1]) / (score_1 + score_2),
            (score_1 * max_entry_1[3] + score_2 * max_entry_2[3]) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        )

    def connection_value(self, ann, paf_i, forward, th, reverse_match=True):
        j1i, j2i = self.skeleton_m1[paf_i]
        if forward:
            jsi = j1i
            directed_paf_field = self.paf_scored.forward[paf_i]
            directed_paf_field_reverse = self.paf_scored.backward[paf_i]
        else:
            jsi = j2i
            directed_paf_field = self.paf_scored.backward[paf_i]
            directed_paf_field_reverse = self.paf_scored.forward[paf_i]
        xyv = ann.data[jsi]
        xy_scale_s = max(0.0, ann.joint_scales[jsi])

        new_xysv = self._grow_connection(xyv[:2], xy_scale_s, directed_paf_field)
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if new_xysv[3] < th:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        if reverse_match:
            reverse_xyv = self._grow_connection(
                new_xysv[:2], xy_scale_t, directed_paf_field_reverse)
            if reverse_xyv[2] < th:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

    def p2p_value(self, source_xyv, source_s, target_xysv, paf_i, forward):
        if forward:
            directed_paf_field = self.paf_scored.forward[paf_i]
        else:
            directed_paf_field = self.paf_scored.backward[paf_i]
        xy_scale_s = max(0.0, source_s)

        # source value
        paf_field = paf_center_s(directed_paf_field, source_xyv[0], source_xyv[1],
                                 sigma=2.0 * xy_scale_s)
        if paf_field.shape[1] == 0:
            return 0.0

        # distances
        d_source = np.linalg.norm(
            ((source_xyv[0],), (source_xyv[1],)) - paf_field[1:3], axis=0)
        d_target = np.linalg.norm(
            ((target_xysv[0],), (target_xysv[1],)) - paf_field[5:7], axis=0)

        # combined value and source distance
        xy_scale_t = max(0.0, target_xysv[2])
        scores = (  # two-tailed cumulative Laplace
            np.exp(-1.0 * d_source / xy_scale_s) *
            np.exp(-1.0 * d_target / xy_scale_t) *
            paf_field[0]
        )
        return np.sqrt(source_xyv[2] * max(scores))

    def _grow(self, ann, th, reverse_match=True):
        frontier = PriorityQueue()
        in_frontier = set()

        def add_to_frontier(start_i):
            for paf_i, forward, end_i in self.by_source[start_i]:
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[paf_i]
                frontier.put((-max_possible_score, None, start_i, end_i, paf_i, forward))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier.qsize():
                entry = frontier.get()
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i, paf_i, forward = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(
                    ann, paf_i, forward, th, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.confidence_scales is not None:
                    score *= self.confidence_scales[paf_i]
                frontier.put((-score, new_xysv, start_i, end_i))

        def confirm(jsi, jti, target_xysv, th=0.2):
            pos = 1 if target_xysv[3] > th else 0
            neg = 0

            for paf_i, forward, source_i in self.by_target[jti]:
                if source_i == jsi:
                    continue

                source_xyv = ann.data[source_i]
                if source_xyv[2] < th:
                    continue
                source_s = ann.joint_scales[source_i]

                v_fixed = self.p2p_value(source_xyv, source_s, target_xysv, paf_i, forward)
                if v_fixed > th:
                    pos += 1
                else:
                    neg += 1

            return pos >= neg

        # seeding the frontier
        for joint_i, v in enumerate(ann.data[:, 2]):
            if v == 0.0:
                continue
            add_to_frontier(joint_i)

        while True:
            entry = frontier_get()
            if entry is None:
                break

            _, new_xysv, jsi, jti = entry
            if ann.data[jti, 2] > 0.0:
                continue
            if self.confirm_connections and not confirm(jsi, jti, new_xysv):
                continue

            ann.data[jti, :2] = new_xysv[:2]
            ann.data[jti, 2] = new_xysv[3]
            ann.joint_scales[jti] = new_xysv[2]
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

            # some joints might still be unfilled
            if np.any(ann.data[:, 2] == 0.0):
                self._flood_fill(ann)

        LOG.debug('complete annotations %.3fs', time.perf_counter() - start)
        return annotations
