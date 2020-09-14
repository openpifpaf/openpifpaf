from collections import defaultdict
import heapq
import logging
import time
from typing import List

import numpy as np

from .generator import Generator
from ...annotation import Annotation
from ..cif_hr import CifHr
from ..cif_seeds import CifSeeds
from ..caf_scored import CafScored
from .. import nms as nms_module
from ..occupancy import Occupancy
from ... import headmeta, visualizer

# pylint: disable=import-error
from ...functional import caf_center_s, grow_connection_blend

LOG = logging.getLogger(__name__)


class CifCaf(Generator):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.0
    nms = True

    def __init__(self,
                 cif_metas: List[headmeta.Cif],
                 caf_metas: List[headmeta.Caf],
                 *,
                 cif_visualizers=None,
                 caf_visualizers=None):
        super().__init__()
        self.cif_metas = cif_metas
        self.caf_metas = caf_metas
        self.skeleton_m1 = np.asarray(self.caf_metas[0].skeleton) - 1
        self.keypoints = cif_metas[0].keypoints
        self.out_skeleton = caf_metas[0].skeleton
        self.confidence_scales = caf_metas[0].decoder_confidence_scales

        self.cif_visualizers = cif_visualizers
        if self.cif_visualizers is None:
            self.cif_visualizers = [visualizer.Cif(meta) for meta in cif_metas]
        self.caf_visualizers = caf_visualizers
        if self.caf_visualizers is None:
            self.caf_visualizers = [visualizer.Caf(meta) for meta in caf_metas]

        if self.nms is True:
            self.nms = nms_module.Keypoints()

        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_target[j2][j1] = (caf_i, True)
            self.by_target[j1][j2] = (caf_i, False)
        self.by_source = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_source[j1][j2] = (caf_i, True)
            self.by_source[j2][j1] = (caf_i, False)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            CifCaf([meta], [meta_next])
            for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
            if (isinstance(meta, headmeta.Cif)
                and isinstance(meta_next, headmeta.Caf))
        ]

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        for vis, meta in zip(self.cif_visualizers, self.cif_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.caf_visualizers, self.caf_metas):
            vis.predicted(fields[meta.head_index])

        cifhr = CifHr().fill(fields, self.cif_metas)
        seeds = CifSeeds(cifhr.accumulated).fill(fields, self.cif_metas)
        caf_scored = CafScored(cifhr.accumulated).fill(fields, self.caf_metas)

        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=4)
        annotations = []

        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma

        for ann in initial_annotations:
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)

        for v, f, x, y, s in seeds.get():
            if occupied.get(f, x, y):
                continue

            ann = Annotation(self.keypoints, self.out_skeleton).add(f, (x, y, v))
            ann.joint_scales[f] = s
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)

        self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        if self.force_complete:
            annotations = self.complete_annotations(cifhr, fields, annotations)

        if self.nms is not None:
            annotations = self.nms.annotations(annotations)

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations

    def connection_value(self, ann, caf_scored, start_i, end_i, *, reverse_match=True):
        caf_i, forward = self.by_source[start_i][end_i]
        caf_f, caf_b = caf_scored.directed(caf_i, forward)
        xyv = ann.data[start_i]
        xy_scale_s = max(0.0, ann.joint_scales[start_i])

        only_max = self.connection_method == 'max'

        new_xysv = grow_connection_blend(
            caf_f, xyv[0], xyv[1], xy_scale_s, only_max)
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        if reverse_match:
            reverse_xyv = grow_connection_blend(
                caf_b, new_xysv[0], new_xysv[1], xy_scale_t, only_max)
            if reverse_xyv[2] == 0.0:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

    @staticmethod
    def p2p_value(source_xyv, caf_scored, source_s, target_xysv, caf_i, forward):
        # TODO move to Cython (see grow_connection_blend)
        caf_f, _ = caf_scored.directed(caf_i, forward)
        xy_scale_s = max(0.0, source_s)

        # source value
        caf_field = caf_center_s(caf_f, source_xyv[0], source_xyv[1],
                                 sigma=2.0 * xy_scale_s)
        if caf_field.shape[1] == 0:
            return 0.0

        # distances
        d_source = np.linalg.norm(
            ((source_xyv[0],), (source_xyv[1],)) - caf_field[1:3], axis=0)
        d_target = np.linalg.norm(
            ((target_xysv[0],), (target_xysv[1],)) - caf_field[5:7], axis=0)

        # combined value and source distance
        xy_scale_t = max(0.0, target_xysv[2])
        sigma_s = 0.5 * xy_scale_s
        sigma_t = 0.5 * xy_scale_t
        scores = (
            np.exp(-0.5 * d_source**2 / sigma_s**2) *
            np.exp(-0.5 * d_target**2 / sigma_t**2) *
            caf_field[0]
        )
        return np.sqrt(source_xyv[2] * max(scores))

    def _grow(self, ann, caf_scored, *, reverse_match=True):
        frontier = []
        in_frontier = set()

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier:
                entry = heapq.heappop(frontier)
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(
                    ann, caf_scored, start_i, end_i, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.greedy:
                    return (-score, new_xysv, start_i, end_i)
                if self.confidence_scales is not None:
                    caf_i, _ = self.by_source[start_i][end_i]
                    score = score * self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-score, new_xysv, start_i, end_i))

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

            ann.data[jti, :2] = new_xysv[:2]
            ann.data[jti, 2] = new_xysv[3]
            ann.joint_scales[jti] = new_xysv[2]
            ann.decoding_order.append(
                (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
            add_to_frontier(jti)

    def _flood_fill(self, ann):
        frontier = []

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                start_xyv = ann.data[start_i].tolist()
                score = xyv[2]
                if self.confidence_scales is not None:
                    score = score * self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-score, end_i, start_xyv, ann.joint_scales[start_i]))

        for start_i, xyv in enumerate(ann.data):
            if xyv[2] == 0.0:
                continue
            add_to_frontier(start_i)

        while frontier:
            _, end_i, xyv, s = heapq.heappop(frontier)
            if ann.data[end_i, 2] > 0.0:
                continue
            ann.data[end_i, :2] = xyv[:2]
            ann.data[end_i, 2] = 0.00001
            ann.joint_scales[end_i] = s
            add_to_frontier(end_i)

    def complete_annotations(self, cifhr, fields, annotations):
        start = time.perf_counter()

        caf_fields = [fields[meta.head_index] for meta in self.caf_metas]
        caf_scored = CafScored(cifhr.accumulated, score_th=0.0001).fill(caf_fields, self.caf_metas)

        for ann in annotations:
            unfilled_mask = ann.data[:, 2] == 0.0
            self._grow(ann, caf_scored, reverse_match=False)
            now_filled_mask = ann.data[:, 2] > 0.0
            updated = np.logical_and(unfilled_mask, now_filled_mask)
            ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])

            # some joints might still be unfilled
            if np.any(ann.data[:, 2] == 0.0):
                self._flood_fill(ann)

        LOG.debug('complete annotations %.3fs', time.perf_counter() - start)
        return annotations
