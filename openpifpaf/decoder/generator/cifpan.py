from collections import defaultdict
import logging
from queue import PriorityQueue
import time

import numpy as np

from .generator import Generator
from ...annotation import Annotation
from ..field_config import FieldConfig
from ..cif_hr import CifHr
from ..cif_seeds import CifSeeds
from ..caf_scored import CafScored
from .. import nms as nms_module
from ..occupancy import Occupancy
from ... import visualizer

import torch

# pylint: disable=import-error
from ...functional import caf_center_s

LOG = logging.getLogger(__name__)


class CifPan(Generator):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.0

    ball = False
    cent = True


    def __init__(self, field_config: FieldConfig, *,
                keypoints,
                #  skeleton,
                 out_skeleton=None,
                 confidence_scales=None,
                 worker_pool=None,
                 nms=True
                ):
        super().__init__(worker_pool)
        if nms is True:
            nms = nms_module.Keypoints()

        self.field_config = field_config

        self.keypoints = keypoints
        # self.skeleton = skeleton
        # self.skeleton_m1 = np.asarray(skeleton) - 1
        self.out_skeleton = out_skeleton
        self.confidence_scales = confidence_scales
        self.nms = nms

        self.timers = defaultdict(float)

        # init by_target and by_source
        # self.by_target = defaultdict(dict)
        # for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
        #     self.by_target[j2][j1] = (caf_i, True)
        #     self.by_target[j1][j2] = (caf_i, False)
        # self.by_source = defaultdict(dict)
        # for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
        #     self.by_source[j1][j2] = (caf_i, True)
        #     self.by_source[j2][j1] = (caf_i, False)

    def __call__(self, fields, initial_annotations=None):
        cif, pan = fields
        semantic, offsets = pan['semantic'], pan['offset']
        print("cif", cif.shape)
        print("semantic", semantic.shape)
        print("offset", offsets.shape)

        Ci, Bi = (17, object()) if self.ball else (17, 18)

        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        # print(self.field_config)
        # if self.field_config.cif_visualizers:
        #     for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
        #         vis.predicted(fields[cif_i])
        # if self.field_config.caf_visualizers:
        #     for vis, caf_i in zip(self.field_config.caf_visualizers, self.field_config.caf_indices):
        #         vis.predicted(fields[caf_i])

        cifhr = CifHr(self.field_config).fill(fields)
        seeds = CifSeeds(cifhr.accumulated, self.field_config).fill(fields)

        # caf_scored = CafScored(cifhr.accumulated, self.field_config, self.skeleton).fill(fields)

        Ñ = None

        def cif_local_max(cif):
            """Use torch for max pooling"""
            cif = torch.tensor(cif)
            cif_m = torch.max_pool2d(cif[None], 7, stride=1, padding=3)[0] == cif
            cif_m &= cif > 0.1
            return np.asarray(cif_m)

        # Get coordinates of keypoints of every type
        # list[K,N_k]
        keypoints_yx = [np.stack(np.nonzero(cif_local_max(cif)), axis=-1)
                        for cif in cifhr.accumulated]

        # Get instance mapping for every pixel
        # keypoints[Ci] tensor[I,2]
        # offsets       tensor[2,H,W]
        # meshgrid      tensor[2,H,W]
        absolute = offsets + np.stack(np.meshgrid(np.arange(offsets.shape[2]),
                                                  np.arange(offsets.shape[1])))
        difference = (absolute[Ñ,:,:,:] -                   # [ ,2,H,W]
                      keypoints_yx[Ci][:,:,Ñ,Ñ]             # [I,2, , ]
                      )

        distances2 = np.square(difference).sum(axis=1)      # [I,H,W]
        instances = distances2.argmin(axis=0)               # [H,W]

        # For each detected keypoints, get its confidence and instance
        centers_fyxv = [
            (Ci, y, x, cifhr.accumulated[Ci,y,x])
            for y, x in keypoints_yx[Ci]
        ]
        if self.ball:
            centers_fyxv += [
                (Bi, y, x, cifhr.accumulated[Bi,y,x])
                for y, x in keypoints_yx[Bi]
            ]
        keypoints_fyxiv = [
            (f, y, x, instances[y,x], cifhr.accumulated[f,y,x])
            for f, kp_yx in enumerate(keypoints_yx[:Ci])
            for y, x in kp_yx
        ]

        annotations = []
        for f, y, x, v in centers_fyxv:
            annotation = Annotation(
                self.keypoints, self.out_skeleton,
                category_id={17:1,18:37}[f]  # center => person, ball center => ball
                )
            annotation.add(f, (x,y,v))
            annotations.append(annotation)

        # Assign keypoints to their instance (least confidence first)
        keypoints_fyxiv.sort(key=lambda x:x[-1])
        for f,y,x,i,v in keypoints_fyxiv:
            annotation = annotations[i]
            annotation.add(f, (x,y,v))

        # semantic      shape [C,H,W]
        classes = semantic.argmin(axis=0)   # [H,W]

        panoptic = classes*1000 + instances
        for i in range(len(annotations)):
            annotation = annotations[i]
            centroid_mask = (classes != 0) & (instances == i)
            annotation.cls = semantic[:,centroid_mask].sum(axis=1).argmax(axis=0)
            annotation.mask = centroid_mask

        # self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        # if self.force_complete:
        #     annotations = self.complete_annotations(cifhr, fields, annotations)

        # if self.nms is not None:
        #     annotations = self.nms.annotations(annotations)

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations

    def _flood_fill(self, ann):
        frontier = PriorityQueue()

        def add_to_frontier(start_i):
            for end_i in self.by_source[start_i].keys():
                if ann.data[end_i, 2] > 0.0:
                    continue
                start_xyv = ann.data[start_i].tolist()
                frontier.put((-xyv[2], end_i, start_xyv, ann.joint_scales[start_i]))

        for start_i, xyv in enumerate(ann.data):
            if xyv[2] == 0.0:
                continue
            add_to_frontier(start_i)

        while frontier.qsize():
            _, end_i, xyv, s = frontier.get()
            if ann.data[end_i, 2] > 0.0:
                continue
            ann.data[end_i, :2] = xyv[:2]
            ann.data[end_i, 2] = 0.00001
            ann.joint_scales[end_i] = s
            add_to_frontier(end_i)

    def complete_annotations(self, cifhr, fields, annotations):
        start = time.perf_counter()

        caf_scored = CafScored(cifhr.accumulated, self.field_config, self.skeleton,
                               score_th=0.0001).fill(fields)

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

    #################### BELOW IS CAF-SPECIFIC ####################

    def _grow(self, ann, caf_scored, *, reverse_match=True):
        frontier = PriorityQueue()
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
                frontier.put((-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier.qsize():
                entry = frontier.get()
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
                    score *= self.confidence_scales[caf_i]
                frontier.put((-score, new_xysv, start_i, end_i))

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

    def _grow_connection(self, xy, xy_scale, caf_field):
        assert len(xy) == 2
        assert caf_field.shape[0] == 9

        # source value
        caf_field = caf_center_s(caf_field, xy[0], xy[1], sigma=2.0 * xy_scale)
        if caf_field.shape[1] == 0:
            return 0, 0, 0, 0

        # source distance
        d = np.linalg.norm(((xy[0],), (xy[1],)) - caf_field[1:3], axis=0)

        # combined value and source distance
        v = caf_field[0]
        sigma = 0.5 * xy_scale
        scores = np.exp(-0.5 * d**2 / sigma**2) * v

        if self.connection_method == 'max':
            return self._target_with_maxscore(caf_field[5:], scores)
        if self.connection_method == 'blend':
            return self._target_with_blend(caf_field[5:], scores)
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
                scores[0] * 0.5,
            )

        sorted_i = np.argsort(scores)
        max_entry_1 = target_coordinates[:, sorted_i[-1]]
        max_entry_2 = target_coordinates[:, sorted_i[-2]]

        score_1 = scores[sorted_i[-1]]
        score_2 = scores[sorted_i[-2]]
        if score_2 < 0.01 or score_2 < 0.5 * score_1:
            return max_entry_1[0], max_entry_1[1], max_entry_1[3], score_1 * 0.5

        # TODO: verify the following three lines have negligible speed impact
        d = np.linalg.norm(max_entry_1[:2] - max_entry_2[:2])
        if d > max_entry_1[3] / 2.0:
            return max_entry_1[0], max_entry_1[1], max_entry_1[3], score_1 * 0.5

        return (
            (score_1 * max_entry_1[0] + score_2 * max_entry_2[0]) / (score_1 + score_2),
            (score_1 * max_entry_1[1] + score_2 * max_entry_2[1]) / (score_1 + score_2),
            (score_1 * max_entry_1[3] + score_2 * max_entry_2[3]) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        )

    def connection_value(self, ann, caf_scored, start_i, end_i, *, reverse_match=True):
        caf_i, forward = self.by_source[start_i][end_i]
        caf_f, caf_b = caf_scored.directed(caf_i, forward)
        xyv = ann.data[start_i]
        xy_scale_s = max(0.0, ann.joint_scales[start_i])

        new_xysv = self._grow_connection(xyv[:2], xy_scale_s, caf_f)
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        if reverse_match:
            reverse_xyv = self._grow_connection(
                new_xysv[:2], xy_scale_t, caf_b)
            if reverse_xyv[2] == 0.0:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

    @staticmethod
    def p2p_value(source_xyv, caf_scored, source_s, target_xysv, caf_i, forward):
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
