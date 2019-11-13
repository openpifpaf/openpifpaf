"""Decoder for pif-paf fields."""

import logging
import time

import numpy as np

from . import generator
from .paf_seeds import PafSeeds
from .utils import normalize_paf7

LOG = logging.getLogger(__name__)


class PafsDijkstra(object):
    force_complete = True
    connection_method = 'blend'
    fixed_b = None
    pif_fixed_scale = None
    paf_th = 0.1

    def __init__(self, stride, *,
                 keypoints,
                 skeleton,
                 paf_index=1,
                 paf_min_distance=0.0,
                 paf_max_distance=None,
                 seed_threshold=0.2,
                 seed_score_scale=1.0,
                 confidence_scales=None,
                 out_skeleton=None,
                 debug_visualizer=None):
        self.strides = stride
        self.paf_indices = paf_index
        self.paf_min_distances = paf_min_distance
        self.paf_max_distances = paf_max_distance
        if not isinstance(self.strides, (list, tuple)):
            self.strides = [self.strides]
            self.paf_indices = [self.paf_indices]
        if not isinstance(self.paf_min_distances, (list, tuple)):
            self.paf_min_distances = [self.paf_min_distances for _ in self.strides]
        if not isinstance(self.paf_max_distances, (list, tuple)):
            self.paf_max_distances = [self.paf_max_distances for _ in self.strides]
        assert len(self.strides) == len(self.paf_indices)
        assert len(self.strides) == len(self.paf_min_distances)
        assert len(self.strides) == len(self.paf_max_distances)

        self.keypoints = keypoints
        self.skeleton = skeleton
        self.out_skeleton = out_skeleton or skeleton

        self.seed_threshold = seed_threshold
        self.seed_score_scale = seed_score_scale
        self.debug_visualizer = debug_visualizer

        self.confidence_scales = confidence_scales

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, paf_i in zip(self.strides, self.paf_indices):
                self.debug_visualizer.paf_raw(fields[paf_i], stride, reg_components=3)

        # confidence scales
        if self.confidence_scales:
            for paf_i in self.paf_indices:
                paf = fields[paf_i]
                cs = np.array(self.confidence_scales, dtype=np.float32).reshape((-1, 1, 1,))
                paf[0] = cs * paf[0]

        # normalize
        normalized_pafs = [normalize_paf7(*fields[paf_i], fixed_b=self.fixed_b)
                           for paf_i in self.paf_indices]
        for p, stride in zip(normalized_pafs, self.strides):
            p[:, :, 1:] *= stride

        # seeds
        seeds = PafSeeds(self.seed_threshold,
                         score_scale=self.seed_score_scale,
                         keypoints=self.keypoints,
                         skeleton=self.skeleton,
                         debug_visualizer=self.debug_visualizer)
        seeds.fill_sequence(normalized_pafs)

        gen = generator.DijkstraPafs(
            normalized_pafs[0],
            seeds,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            paf_th=self.paf_th,
            keypoints=self.keypoints,
            skeleton=self.skeleton,
            out_skeleton=self.out_skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        if self.force_complete:
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
