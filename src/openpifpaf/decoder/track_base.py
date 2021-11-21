import argparse
from typing import List

import numpy as np

from .. import annotation
from .decoder import Decoder
from .track_annotation import TrackAnnotation
from ..signal import Signal


class TrackBase(Decoder):
    single_pose_threshold = 0.3
    multi_pose_threshold = 0.2
    multi_pose_n = 3
    minimum_threshold = 0.1
    simplify_good_ids = True
    track_visualizer = None

    def __init__(self):
        super().__init__()

        self.active: List[TrackAnnotation] = []
        self.frame_number = 0

        self.simplified_track_id_map = {}
        self.simplified_last_track_id = 0

        Signal.subscribe('eval_reset', self.reset)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('Decoder for tracking')
        group.add_argument('--tr-single-pose-threshold',
                           default=cls.single_pose_threshold, type=float,
                           help='Single-pose threshold for tracking.')
        group.add_argument('--tr-multi-pose-threshold',
                           default=cls.multi_pose_threshold, type=float,
                           help='multi-pose threshold for tracking.')
        group.add_argument('--tr-multi-pose-n',
                           default=cls.multi_pose_n, type=float,
                           help='multi-pose n for tracking.')
        group.add_argument('--tr-minimum-threshold',
                           default=cls.minimum_threshold, type=float,
                           help='minimum-pose threshold for tracking.')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.single_pose_threshold = args.tr_single_pose_threshold
        cls.multi_pose_threshold = args.tr_multi_pose_threshold
        cls.multi_pose_n = args.tr_multi_pose_n
        cls.minimum_threshold = args.tr_minimum_threshold

    @classmethod
    def factory(cls, head_metas) -> List['Decoder']:
        """Create instances of an implementation."""
        raise NotImplementedError

    def __call__(self, fields, *, initial_annotations=None) -> List[annotation.Base]:
        """For single image, from fields to annotations."""
        raise NotImplementedError

    def simplify_ids(self, ids):
        out = []
        for id_ in ids:
            if id_ not in self.simplified_track_id_map:
                self.simplified_last_track_id += 1
                self.simplified_track_id_map[id_] = self.simplified_last_track_id

            out.append(self.simplified_track_id_map[id_])
        return out

    def reset(self):
        self.active = []
        self.frame_number = 0

        self.simplified_track_id_map = {}
        self.simplified_last_track_id = 0

    def prune_active(self, frame_number):
        self.active = [t for t in self.active
                       if frame_number - t.frame_pose[-1][0] <= 33]
        self.active = [t for t in self.active
                       if frame_number - t.frame_pose[-1][0] == 1 or len(t.frame_pose) > 2]

    def annotations(self, frame_number):
        tracks = [t for t in self.active if t.frame_pose[-1][0] == frame_number]
        tracks = [t for t in tracks if self.track_is_good(t, frame_number)]
        if not tracks:
            return []

        ids = [t.id_ for t in tracks]
        if self.simplify_good_ids:
            ids = self.simplify_ids(ids)
        annotations = [t.frame_pose[-1][1] for t in tracks]
        for ann, id_ in zip(annotations, ids):
            ann.id_ = id_
        return annotations

    def tag_ignore_region(self, frame_number, gt_anns):
        pose_annotations = [track.frame_pose[-1][1]
                            for track in self.active
                            if track.frame_pose[-1][0] == frame_number]
        crowd_annotations = [a for a in gt_anns if a['iscrowd']]

        def point_in_polygon(x, y, poly_x, poly_y):
            inside = False
            for x1, x2, y1, y2 in zip(poly_x[:-1], poly_x[1:], poly_y[:-1], poly_y[1:]):
                if min(y1, y2) > y or max(y1, y2) < y:
                    continue
                lx = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                if lx < x:
                    inside = not inside
            return inside

        def pa_in_ca(pose_annotation, crowd_annotation):
            pose = pose_annotation.data
            poly = crowd_annotation['keypoints'][:, :2].tolist()
            # close polygon
            poly.append(poly[0])
            xs = [x for x, _ in poly]
            ys = [y for _, y in poly]

            kp_order = np.argsort(pose[:, 2])[::-1]
            if all(point_in_polygon(kp[0], kp[1], xs, ys)
                   for kp in pose[kp_order[:3]] if kp[2] > 0.05):
                return True

            return False

        for pa in pose_annotations:
            pa.ignore_region = any(pa_in_ca(pa, ca) for ca in crowd_annotations)

    def track_is_viable(self, track, frame_number):
        if frame_number > track.frame_pose[-1][0] + 33:
            return False

        if any(track.pose_score(frame_number - i) > self.multi_pose_threshold for i in range(33)):
            return True

        return False

    def track_is_good(self, track, frame_number):
        for i in range(4):
            pose = track.pose(frame_number - i)
            if pose is None:
                continue
            if getattr(pose, 'ignore_region', False):
                return False

        if not self.track_is_viable(track, frame_number):
            return False

        if all(track.pose_score(frame_number - i) < self.single_pose_threshold
               for i in range(6)) and \
           sum(1
               for i in range(6)
               if track.pose_score(frame_number - i) > self.multi_pose_threshold
               ) < self.multi_pose_n:
            return False

        assert self.minimum_threshold >= 0.0  # make sure to return False when pose is None
        if track.pose_score(frame_number) <= self.minimum_threshold:
            return False

        return True
