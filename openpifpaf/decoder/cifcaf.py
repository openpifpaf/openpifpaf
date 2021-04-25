import argparse
from collections import defaultdict
import logging
import time
from typing import List

import numpy as np
import torch

from .decoder import Decoder
from ..annotation import Annotation
from . import utils
from .. import headmeta, visualizer

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
    nms_before_force_complete = False
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

        # init by_target and by_source
        self.by_target = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_target[j2][j1] = (caf_i, True)
            self.by_target[j1][j2] = (caf_i, False)
        self.by_source = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_source[j1][j2] = (caf_i, True)
            self.by_source[j2][j1] = (caf_i, False)

        self.cpp_decoder = torch.classes.openpifpaf_decoder.CifCaf(
            len(self.keypoints),
            torch.from_numpy(self.skeleton_m1),
        )

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        CppCifCaf = torch.classes.openpifpaf_decoder.CifCaf  # pylint: disable=invalid-name

        group = parser.add_argument_group('CifCaf decoder')
        assert not CppCifCaf.get_force_complete()
        group.add_argument('--force-complete-pose',
                           default=False, action='store_true')
        group.add_argument('--force-complete-caf-th', type=float,
                           default=CppCifCaf.get_force_complete_caf_th(),
                           help='CAF threshold for force complete. Set to -1 to deactivate.')
        assert not cls.nms_before_force_complete
        group.add_argument('--nms-before-force-complete', default=False, action='store_true',
                           help='run an additional NMS before completing poses')

        assert utils.NMSKeypoints.get_keypoint_threshold() == CppCifCaf.get_keypoint_threshold()
        group.add_argument('--keypoint-threshold', type=float,
                           default=CppCifCaf.get_keypoint_threshold(),
                           help='filter keypoints by score')
        group.add_argument('--keypoint-threshold-rel', type=float,
                           default=CppCifCaf.get_keypoint_threshold_rel(),
                           help='filter keypoint connections by relative score')

        assert not CppCifCaf.get_greedy()
        group.add_argument('--greedy', default=False, action='store_true',
                           help='greedy decoding')
        group.add_argument('--connection-method',
                           default=cls.connection_method,
                           choices=('max', 'blend'),
                           help='connection method to use, max is faster')
        group.add_argument('--dense-connections', nargs='?', type=float,
                           default=0.0, const=1.0)

        assert CppCifCaf.get_reverse_match()
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
        CppCifCaf = torch.classes.openpifpaf_decoder.CifCaf  # pylint: disable=invalid-name

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

        cls.nms_before_force_complete = args.nms_before_force_complete
        utils.NMSKeypoints.set_keypoint_threshold(keypoint_threshold_nms)

        CppCifCaf.set_force_complete(args.force_complete_pose)
        CppCifCaf.set_force_complete_caf_th(args.force_complete_caf_th)
        CppCifCaf.set_keypoint_threshold(args.keypoint_threshold)
        CppCifCaf.set_keypoint_threshold_rel(args.keypoint_threshold_rel)

        CppCifCaf.set_greedy(args.greedy)
        cls.connection_method = args.connection_method
        cls.dense_coupling = args.dense_connections

        cls.reverse_match = args.reverse_match
        # TODO utils.CifSeeds.ablation_nms = args.ablation_cifseeds_nms
        # TODO utils.CifSeeds.ablation_no_rescore = args.ablation_cifseeds_no_rescore
        # TODO utils.CafScored.ablation_no_rescore = args.ablation_caf_no_rescore
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
        annotations = self.cpp_decoder.call(
            fields[self.cif_metas[0].head_index],
            8,  # self.cif_metas[0].stride,
            fields[self.caf_metas[0].head_index],
            8,  # self.caf_metas[0].stride,
        )
        LOG.debug('cpp annotations = %d (%.1fms)',
                  len(annotations),
                  (time.perf_counter() - start) * 1000.0)

        annotations_py = []
        for ann_data in annotations:
            ann = Annotation(self.keypoints,
                             self.out_skeleton,
                             score_weights=self.score_weights)
            ann.data[:, :2] = ann_data[:, 1:3]
            ann.data[:, 2] = ann_data[:, 0]
            ann.joint_scales[:] = ann_data[:, 3]
            annotations_py.append(ann)
        print([np.sum(ann.data[:, 2] > 0.0) for ann in annotations_py])
        return annotations_py
