import argparse
import logging
import time

import numpy as np
try:
    import scipy.optimize
except ImportError:
    scipy = None  # pylint: disable=invalid-name

from .. import headmeta
from .cifcaf import CifCaf
from .track_annotation import TrackAnnotation
from .track_base import TrackBase
from . import pose_distance

LOG = logging.getLogger(__name__)


class PoseSimilarity(TrackBase):
    distance_type = pose_distance.Euclidean

    def __init__(self, cif_meta: headmeta.Cif, caf_meta: headmeta.Caf, *, pose_generator=None):
        super().__init__()
        self.cif_meta = cif_meta
        self.caf_meta = caf_meta

        # prefer decoders with more keypoints and associations
        self.priority = -10.0
        self.priority += cif_meta.n_fields / 1000.0
        self.priority += caf_meta.n_fields / 1000.0

        self.distance_function = self.distance_type()
        self.distance_function.valid_keypoints = [
            i
            for i, kp in enumerate(cif_meta.keypoints)
            if kp not in (('left_ear', 'right_ear') if cif_meta.dataset == 'posetrack2018' else [])
        ]
        LOG.debug('valid keypoints = %s', self.distance_function.valid_keypoints)
        self.distance_function.sigmas = np.asarray(cif_meta.sigmas)

        self.pose_generator = pose_generator or CifCaf([cif_meta], [caf_meta])

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('PoseSimilarity')
        assert cls.distance_type == pose_distance.Euclidean
        group.add_argument('--posesimilarity-distance', default='euclidean',
                           choices=('crafted', 'euclidean', 'euclidean4', 'oks'))
        group.add_argument('--posesimilarity-oks-inflate',
                           default=pose_distance.Oks.inflate, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        if args.posesimilarity_distance == 'euclidean':
            cls.distance_type = pose_distance.Euclidean
        elif args.posesimilarity_distance == 'euclidean4':
            cls.distance_type = lambda _: pose_distance.Euclidean(track_frames=[-1, -4, -8, -12])
        elif args.posesimilarity_distance == 'oks':
            cls.distance_type = pose_distance.Oks
        elif args.posesimilarity_distance == 'crafted':
            cls.distance_type = pose_distance.Crafted
        else:
            raise Exception('distance function type not known')

        pose_distance.Oks.inflate = args.posesimilarity_oks_inflate

    @classmethod
    def factory(cls, head_metas):
        if len(head_metas) < 2:
            return []
        return [
            cls(cif_meta, caf_meta)
            for cif_meta, caf_meta
            in zip(head_metas, head_metas[1:])
            if (isinstance(cif_meta, (headmeta.TSingleImageCif, headmeta.Cif))
                and isinstance(caf_meta, (headmeta.TSingleImageCaf, headmeta.Caf)))
        ]

    def __call__(self, fields, *, initial_annotations=None):
        self.frame_number += 1
        start = time.perf_counter()

        self.prune_active(self.frame_number)

        pose_start = time.perf_counter()
        pose_annotations = self.pose_generator(fields)
        LOG.debug('pose time = %.3fs', time.perf_counter() - pose_start)

        cost_start = time.perf_counter()
        cost = np.full((len(self.active) * 2, len(pose_annotations)), 1000.0)
        for track_i, track in enumerate(self.active):
            for pose_i, pose in enumerate(pose_annotations):
                cost[track_i, pose_i] = self.distance_function(
                    self.frame_number, pose, track, self.track_is_good(track, self.frame_number))

                # option to loose track (e.g. occlusion)
                cost[track_i + len(self.active), pose_i] = 100.0
        LOG.debug('cost time = %.3fs', time.perf_counter() - cost_start)

        track_indices, pose_indices = scipy.optimize.linear_sum_assignment(cost)
        matched_poses = set()
        for track_i, pose_i in zip(track_indices, pose_indices):
            # was track lost?
            if track_i >= len(self.active):
                continue

            pose = pose_annotations[pose_i]
            track = self.active[track_i]

            track.add(self.frame_number, pose)
            matched_poses.add(pose)

        for new_pose in pose_annotations:
            if new_pose in matched_poses:
                continue
            self.active.append(TrackAnnotation().add(self.frame_number, new_pose))

        # tag ignore regions TODO
        # if self.gt_anns:
        #     self.tag_ignore_region(self.frame_number, self.gt_anns)

        # pruning lost tracks
        self.active = [t for t in self.active if self.track_is_viable(t, self.frame_number)]

        LOG.info('active tracks = %d, good = %d, track ids = %s',
                 len(self.active),
                 len([t for t in self.active if self.track_is_good(t, self.frame_number)]),
                 [self.simplified_track_id_map.get(t.id_, t.id_)
                  for t in self.active])

        if self.track_visualizer:
            self.track_visualizer.predicted(
                self.frame_number,
                [t for t in self.active if self.track_is_good(t, self.frame_number)],
            )

        LOG.debug('track time: %.3fs', time.perf_counter() - start)
        return self.annotations(self.frame_number)
