from collections import defaultdict
import logging
import time

import numpy as np

from ..annotation import Annotation
from ..utils import scalar_square_add_single

# pylint: disable=import-error
from ...functional import paf_center, scalar_value, scalar_nonzero, weiszfeld_nd

LOG = logging.getLogger(__name__)


class Greedy(object):
    def __init__(self, pifhr, paf_scored, seeds, *,
                 seed_threshold,
                 connection_method,
                 paf_nn,
                 paf_th,
                 keypoints,
                 skeleton,
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

        self.debug_visualizer = debug_visualizer
        self.timers = defaultdict(float)

        if self.debug_visualizer:
            self.debug_visualizer.pifhr(self.pifhr.targets)

    def annotations(self, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        occupied = np.zeros(self.pifhr.scales.shape, dtype=np.uint8)
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
            if ann.joint_scales is None:
                ann.fill_joint_scales(self.pifhr.scales)
            self._grow(ann, self.paf_th)
            ann.fill_joint_scales(self.pifhr.scales)
            annotations.append(ann)
            mark_occupied(ann)

        for v, f, x, y, _ in self.seeds.get():
            if scalar_nonzero(occupied[f], x, y):
                continue

            ann = Annotation(self.keypoints, self.skeleton).add(f, (x, y, v))
            self._grow(ann, self.paf_th)
            ann.fill_joint_scales(self.pifhr.scales)
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
        paf_field = paf_center(paf_field, xy[0], xy[1], sigma=5.0 * xy_scale)
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
            return self._target_with_maxscore(paf_field[4:6], scores)
        if self.connection_method == 'blend':
            return self._target_with_blend(paf_field[4:6], scores)
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
        if score_2 < 0.01 or score_2 < 0.5 * score_1:
            return max_entry_1[0], max_entry_1[1], score_1

        return (
            (score_1 * max_entry_1[0] + score_2 * max_entry_2[0]) / (score_1 + score_2),
            (score_1 * max_entry_1[1] + score_2 * max_entry_2[1]) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        )

    def _grow(self, ann, th, reverse_match=True):
        for _, i, forward, j1i, j2i in ann.frontier_iter():
            if forward:
                jsi, jti = j1i, j2i
                directed_paf_field = self.paf_scored.forward[i]
                directed_paf_field_reverse = self.paf_scored.backward[i]
            else:
                jsi, jti = j2i, j1i
                directed_paf_field = self.paf_scored.backward[i]
                directed_paf_field_reverse = self.paf_scored.forward[i]
            xyv = ann.data[jsi]
            xy_scale_s = max(
                4.0,
                scalar_value(self.pifhr.scales[jsi], xyv[0], xyv[1])
            )

            new_xyv = self._grow_connection(xyv[:2], xy_scale_s, directed_paf_field)
            if new_xyv[2] < th:
                continue
            xy_scale_t = max(
                4.0,
                scalar_value(self.pifhr.scales[jti], new_xyv[0], new_xyv[1])
            )

            # reverse match
            if reverse_match:
                reverse_xyv = self._grow_connection(
                    new_xyv[:2], xy_scale_t, directed_paf_field_reverse)
                if reverse_xyv[2] < th:
                    continue
                if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                    continue

            new_xyv = (new_xyv[0], new_xyv[1], np.sqrt(new_xyv[2] * xyv[2]))  # geometric mean
            if new_xyv[2] > ann.data[jti, 2]:
                ann.data[jti] = new_xyv
                ann.decoding_order.append(
                    (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))

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
            ann.fill_joint_scales(self.pifhr.scales)

            # some joints might still be unfilled
            if np.any(ann.data[:, 2] == 0.0):
                self._flood_fill(ann)

        LOG.debug('complete annotations %.3fs', time.perf_counter() - start)
        return annotations
