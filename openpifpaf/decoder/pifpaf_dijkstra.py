"""Decoder for pif-paf fields."""

import logging
import time

import numpy as np

from . import generator
from .paf_scored import PafScored
from .pif_hr import PifHr
from .pif_seeds import PifSeeds
from .utils import normalize_pif, normalize_paf

LOG = logging.getLogger(__name__)


class PifPafDijkstra(object):
    force_complete = True
    connection_method = 'blend'
    fixed_b = None
    pif_fixed_scale = None
    paf_th = 0.1

    def __init__(self, stride, *,
                 keypoints,
                 skeleton,
                 pif_index=0, paf_index=1,
                 pif_min_scale=0.0,
                 paf_min_distance=0.0,
                 paf_max_distance=None,
                 seed_threshold=0.2,
                 confidence_scales=None,
                 debug_visualizer=None):
        self.strides = stride
        self.pif_indices = pif_index
        self.paf_indices = paf_index
        self.pif_min_scales = pif_min_scale
        self.paf_min_distances = paf_min_distance
        self.paf_max_distances = paf_max_distance
        if not isinstance(self.strides, (list, tuple)):
            self.strides = [self.strides]
            self.pif_indices = [self.pif_indices]
            self.paf_indices = [self.paf_indices]
        if not isinstance(self.pif_min_scales, (list, tuple)):
            self.pif_min_scales = [self.pif_min_scales for _ in self.strides]
        if not isinstance(self.paf_min_distances, (list, tuple)):
            self.paf_min_distances = [self.paf_min_distances for _ in self.strides]
        if not isinstance(self.paf_max_distances, (list, tuple)):
            self.paf_max_distances = [self.paf_max_distances for _ in self.strides]
        assert len(self.strides) == len(self.pif_indices)
        assert len(self.strides) == len(self.paf_indices)
        assert len(self.strides) == len(self.pif_min_scales)
        assert len(self.strides) == len(self.paf_min_distances)
        assert len(self.strides) == len(self.paf_max_distances)

        self.keypoints = keypoints
        self.skeleton = skeleton

        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

        self.confidence_scales = confidence_scales

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, pif_i in zip(self.strides, self.pif_indices):
                self.debug_visualizer.pif_raw(fields[pif_i], stride)
            for stride, paf_i in zip(self.strides, self.paf_indices):
                self.debug_visualizer.paf_raw(fields[paf_i], stride, reg_components=3)

        # confidence scales
        if self.confidence_scales:
            for paf_i in self.paf_indices:
                paf = fields[paf_i]
                cs = np.array(self.confidence_scales, dtype=np.float32).reshape((-1, 1, 1,))
                paf[0] = cs * paf[0]

        # normalize
        normalized_pifs = [normalize_pif(*fields[pif_i], fixed_scale=self.pif_fixed_scale)
                           for pif_i in self.pif_indices]
        normalized_pafs = [normalize_paf(*fields[paf_i], fixed_b=self.fixed_b)
                           for paf_i in self.paf_indices]

        # pif hr
        pifhr = PifHr(self.pif_nn)
        pifhr.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)

        # seeds
        seeds = PifSeeds(pifhr.target_accumulator, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer)
        seeds.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)

        # paf_scored
        paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=self.paf_th)
        paf_scored.fill_sequence(
            normalized_pafs, self.strides, self.paf_min_distances, self.paf_max_distances)

        gen = generator.Dijkstra(
            pifhr, paf_scored, seeds,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            paf_nn=self.paf_nn,
            paf_th=self.paf_th,
            keypoints=self.keypoints,
            skeleton=self.skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        if self.force_complete:
            gen.paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=0.0001)
            gen.paf_scored.fill_sequence(
                normalized_pafs, self.strides, self.paf_min_distances, self.paf_max_distances)
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
