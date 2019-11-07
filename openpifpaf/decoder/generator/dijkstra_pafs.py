from collections import defaultdict
import logging
from queue import PriorityQueue
import time

import numpy as np

from ..annotation import Annotation
from ..utils import scalar_square_add_single

# pylint: disable=import-error
from ...functional import paf_center_s, scalar_nonzero, weiszfeld_nd

LOG = logging.getLogger(__name__)


class DijkstraPafs(object):
    def __init__(self, paf, seeds, *,
                 seed_threshold,
                 connection_method,
                 paf_nn,
                 paf_th,
                 keypoints,
                 skeleton,
                 out_skeleton=None,
                 debug_visualizer=None):
        self.paf = paf
        self.seeds = seeds

        self.seed_threshold = seed_threshold
        self.connection_method = connection_method
        self.paf_nn = paf_nn
        self.paf_th = paf_th

        self.keypoints = keypoints
        self.skeleton = skeleton
        self.skeleton_m1 = np.asarray(skeleton) - 1
        self.out_skeleton = out_skeleton or skeleton

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

    def annotations(self, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        # TODO stride to replace 8
        hr_shape = (len(self.keypoints), self.paf.shape[3] * 8, self.paf.shape[4] * 8)
        occupied = np.zeros(hr_shape, dtype=np.uint8)
        annotations = []

        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                scalar_square_add_single(occupied[joint_i],
                                         xyv[0],
                                         xyv[1],
                                         max(4.0, width),
                                         1)

        for ann in initial_annotations:
            self._grow(ann, self.paf_th)
            annotations.append(ann)
            mark_occupied(ann)

        for kps4 in self.seeds.get():
            if any(scalar_nonzero(occupied[f], x, y)
                   for f, (x, y, s, v) in enumerate(kps4)
                   if v > 0.0):
                continue

            ann = Annotation(self.keypoints, self.out_skeleton)
            ann.data[:, 0:2] = kps4[:, 0:2]
            ann.data[:, 2] = kps4[:, 3]
            ann.joint_scales = kps4[:, 2]
            self._grow(ann, self.paf_th)
            annotations.append(ann)
            mark_occupied(ann)

        if self.debug_visualizer:
            LOG.debug('occupied field 0')
            self.debug_visualizer.occupied(occupied[0])

        LOG.debug('keypoint sets %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations

    def _grow_connection(self, xy, xy_scale, paf_field):
        assert len(xy) == 2
        assert paf_field.shape[0] == 2
        assert paf_field.shape[1] == 5
        paf_field = np.reshape(paf_field, (2, 5, -1))
        paf_field = paf_field[:, [0, 1, 2, 4]]

        # source value
        paf_field = paf_center_s(paf_field, xy[0], xy[1], sigma=5.0 * xy_scale)
        if paf_field.shape[2] == 0:
            return 0, 0, 0, 0

        # source distance
        d = np.linalg.norm(((xy[0],), (xy[1],)) - paf_field[0, 1:3], axis=0)

        # combined value and source distance
        v = paf_field[0, 0]
        scores = np.exp(-1.0 * d / xy_scale) * v  # two-tailed cumulative Laplace

        if self.connection_method == 'max':
            return self._target_with_maxscore(paf_field[1, 1:4], scores)
        if self.connection_method == 'blend':
            return self._target_with_blend(paf_field[1, 1:4], scores)
        raise Exception('connection method not known')

    @staticmethod
    def _target_with_maxscore(target_coordinates, scores):
        assert target_coordinates.shape[1] == scores.shape[0]

        max_i = np.argmax(scores)
        max_entry = target_coordinates[:, max_i]

        score = scores[max_i]
        return max_entry[0], max_entry[1], max_entry[2], score

    @staticmethod
    def _target_with_blend(target_coordinates, scores):
        """Blending the top two candidates with a weighted average.

        Similar to the post processing step in
        "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs".
        """
        assert target_coordinates.shape[1] == len(scores)
        if len(scores) == 1:
            return target_coordinates[0, 0], target_coordinates[1, 0], target_coordinates[2, 0], scores[0]

        sorted_i = np.argsort(scores)
        max_entry_1 = target_coordinates[:, sorted_i[-1]]
        max_entry_2 = target_coordinates[:, sorted_i[-2]]

        score_1 = scores[sorted_i[-1]]
        score_2 = scores[sorted_i[-2]]
        if score_2 < 0.01 or score_2 < 0.5 * score_1:
            return max_entry_1[0], max_entry_1[1], max_entry_1[2], score_1

        return (
            (score_1 * max_entry_1[0] + score_2 * max_entry_2[0]) / (score_1 + score_2),
            (score_1 * max_entry_1[1] + score_2 * max_entry_2[1]) / (score_1 + score_2),
            (score_1 * max_entry_1[2] + score_2 * max_entry_2[2]) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        )

    def connection_value(self, ann, paf_i, forward, th, reverse_match=True):
        j1i, j2i = self.skeleton_m1[paf_i]
        if forward:
            jsi = j1i
            directed_paf_field = self.paf[paf_i]
            directed_paf_field_reverse = self.paf[paf_i, ::-1]
        else:
            jsi = j2i
            directed_paf_field = self.paf[paf_i, ::-1]
            directed_paf_field_reverse = self.paf[paf_i]
        xyv = ann.data[jsi]
        xy_scale_s = max(
            8.0,
            ann.joint_scales[jsi]
        )

        new_xysv = self._grow_connection(xyv[:2], xy_scale_s, directed_paf_field)
        if new_xysv[3] < th:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(
            8.0,
            new_xysv[2]
        )

        # reverse match
        if reverse_match:
            reverse_xyv = self._grow_connection(
                new_xysv[:2], xy_scale_t, directed_paf_field_reverse)
            if reverse_xyv[2] < th:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], np.sqrt(new_xysv[3] * xyv[2]))  # geometric mean

    def _grow(self, ann, th, reverse_match=True):
        frontier = PriorityQueue()
        evaluated_connections = set()

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

                new_xysv = self.connection_value(
                    ann, paf_i, forward, th, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                frontier.put((-new_xysv[3], new_xysv, start_i, end_i))
                evaluated_connections.add((start_i, end_i))

        # seeding the frontier
        for joint_i, v in enumerate(ann.data[:, 2]):
            if v == 0.0:
                continue
            add_to_frontier(joint_i)

        while frontier.qsize():
            _, new_xysv, jsi, jti = frontier.get()
            if ann.data[jti, 2] > 0.0:
                continue

            ann.data[jti, 0:2] = new_xysv[0:2]
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
