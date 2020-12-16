import argparse
from collections import defaultdict
import heapq
import logging
import time
from typing import List

import numpy as np

from .decoder import Decoder
from ..annotation import Annotation
from . import utils
from .. import headmeta, visualizer

# pylint: disable=import-error
from ..functional import caf_center_s, grow_connection_blend

LOG = logging.getLogger(__name__)


class DenseAdapter:
    def __init__(self, cif_meta, caf_meta, dense_caf_meta):
        self.cif_meta = cif_meta
        self.caf_meta = caf_meta
        self.dense_caf_meta = dense_caf_meta

        # overwrite confidence scale
        self.dense_caf_meta.confidence_scales = [
            CifCaf.dense_coupling for _ in self.dense_caf_meta.skeleton
        ]

        concatenated_caf_meta = headmeta.Caf.concatenate(
            [caf_meta, dense_caf_meta])
        self.cifcaf = CifCaf([cif_meta], [concatenated_caf_meta])

    @classmethod
    def factory(cls, head_metas):
        if len(head_metas) < 3:
            return []
        return [
            DenseAdapter(cif_meta, caf_meta, dense_meta)
            for cif_meta, caf_meta, dense_meta in zip(head_metas, head_metas[1:], head_metas[2:])
            if (isinstance(cif_meta, headmeta.Cif)
                and isinstance(caf_meta, headmeta.Caf)
                and isinstance(dense_meta, headmeta.Caf))
        ]

    def __call__(self, fields, initial_annotations=None):
        cifcaf_fields = [
            fields[self.cif_meta.head_index],
            np.concatenate([
                fields[self.caf_meta.head_index],
                fields[self.dense_caf_meta.head_index],
            ], axis=0)
        ]
        return self.cifcaf(cifcaf_fields)


class CifCaf(Decoder):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.15
    keypoint_threshold_rel = 0.5
    nms = utils.nms.Keypoints()
    dense_coupling = 0.0

    reverse_match = True

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
        self.score_weights = cif_metas[0].score_weights
        self.out_skeleton = caf_metas[0].skeleton
        self.confidence_scales = caf_metas[0].decoder_confidence_scales

        self.cif_visualizers = cif_visualizers
        if self.cif_visualizers is None:
            self.cif_visualizers = [visualizer.Cif(meta) for meta in cif_metas]
        self.caf_visualizers = caf_visualizers
        if self.caf_visualizers is None:
            self.caf_visualizers = [visualizer.Caf(meta) for meta in caf_metas]

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
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('CifCaf decoder')
        assert not cls.force_complete
        group.add_argument('--force-complete-pose',
                           default=False, action='store_true')

        assert utils.nms.Keypoints.keypoint_threshold == cls.keypoint_threshold
        group.add_argument('--keypoint-threshold', type=float,
                           default=cls.keypoint_threshold,
                           help='filter keypoints by score')
        group.add_argument('--keypoint-threshold-rel', type=float,
                           default=cls.keypoint_threshold_rel,
                           help='filter keypoint connections by relative score')

        assert not cls.greedy
        group.add_argument('--greedy', default=False, action='store_true',
                           help='greedy decoding')
        group.add_argument('--connection-method',
                           default=cls.connection_method,
                           choices=('max', 'blend'),
                           help='connection method to use, max is faster')
        group.add_argument('--dense-connections', nargs='?', type=float,
                           default=0.0, const=1.0)

        assert cls.reverse_match
        group.add_argument('--no-reverse-match',
                           default=True, dest='reverse_match', action='store_false')
        group.add_argument('--ablation-cifseeds-nms',
                           default=False, action='store_true')
        group.add_argument('--ablation-cifseeds-no-rescore',
                           default=False, action='store_true')
        group.add_argument('--ablation-caf-no-rescore',
                           default=False, action='store_true')
        group.add_argument('--ablation-independent-kp',
                           default=False, action='store_true')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        # force complete
        keypoint_threshold_nms = args.keypoint_threshold
        if args.force_complete_pose:
            if not args.ablation_independent_kp:
                args.keypoint_threshold = 0.0
            args.keypoint_threshold_rel = 0.0
            keypoint_threshold_nms = 0.0
        # check consistency
        if args.seed_threshold < args.keypoint_threshold:
            LOG.warning(
                'consistency: decreasing keypoint threshold to seed threshold of %f',
                args.seed_threshold,
            )
            args.keypoint_threshold = args.seed_threshold

        cls.force_complete = args.force_complete_pose
        cls.keypoint_threshold = args.keypoint_threshold
        utils.nms.Keypoints.keypoint_threshold = keypoint_threshold_nms
        cls.keypoint_threshold_rel = args.keypoint_threshold_rel

        cls.greedy = args.greedy
        cls.connection_method = args.connection_method
        cls.dense_coupling = args.dense_connections

        cls.reverse_match = args.reverse_match
        utils.CifSeeds.ablation_nms = args.ablation_cifseeds_nms
        utils.CifSeeds.ablation_no_rescore = args.ablation_cifseeds_no_rescore
        utils.CafScored.ablation_no_rescore = args.ablation_caf_no_rescore
        if args.ablation_cifseeds_no_rescore and args.ablation_caf_no_rescore:
            utils.CifHr.ablation_skip = True

    @classmethod
    def factory(cls, head_metas):
        if cls.dense_coupling:
            return DenseAdapter.factory(head_metas)
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

        cifhr = utils.CifHr().fill(fields, self.cif_metas)
        seeds = utils.CifSeeds(cifhr.accumulated).fill(fields, self.cif_metas)
        caf_scored = utils.CafScored(cifhr.accumulated).fill(fields, self.caf_metas)

        occupied = utils.Occupancy(cifhr.accumulated.shape, 2, min_scale=4)
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

            ann = Annotation(self.keypoints,
                             self.out_skeleton,
                             score_weights=self.score_weights
                             ).add(f, (x, y, v))
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
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if keypoint_score / max(0.01, xyv[2]) < self.keypoint_threshold_rel:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        if self.reverse_match and reverse_match:
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
            np.exp(-0.5 * d_source**2 / sigma_s**2)
            * np.exp(-0.5 * d_target**2 / sigma_t**2)
            * caf_field[0]
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

        caf_scored = utils.CafScored(cifhr.accumulated, score_th=0.001).fill(
            fields, self.caf_metas)
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
