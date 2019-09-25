"""Decoder for pif-paf fields."""

import logging
import time

import numpy as np

from . import generator
from .decoder import Decoder
from .paf_scored import PafScored
from .pif_hr import PifHr
from .pif_seeds import PifSeeds
from .utils import normalize_pif, normalize_paf

LOG = logging.getLogger(__name__)


class PifPafDijkstra(Decoder):
    force_complete = True
    connection_method = 'max'
    fixed_b = None
    pif_fixed_scale = None
    paf_th = 0.1

    def __init__(self, stride, *,
                 skeleton,
                 pif_index=0, paf_index=1,
                 pif_min_scale=0.0,
                 paf_min_distance=0.0,
                 seed_threshold=0.2,
                 confidence_scales=None,
                 debug_visualizer=None):
        self.strides = stride
        self.pif_indeces = pif_index
        self.paf_indeces = paf_index
        self.pif_min_scales = pif_min_scale
        self.paf_min_distances = paf_min_distance
        if not isinstance(self.strides, (list, tuple)):
            self.strides = [self.strides]
            self.pif_indeces = [self.pif_indeces]
            self.paf_indeces = [self.paf_indeces]
        if not isinstance(self.pif_min_scales, (list, tuple)):
            self.pif_min_scales = [self.pif_min_scales for _ in self.strides]
        if not isinstance(self.paf_min_distances, (list, tuple)):
            self.paf_min_distances = [self.paf_min_distances for _ in self.strides]

        self.skeleton = skeleton

        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

        self.confidence_scales = confidence_scales

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, pif_i in zip(self.strides, self.pif_indeces):
                self.debug_visualizer.pif_raw(fields[pif_i], stride)
            for stride, paf_i in zip(self.strides, self.paf_indeces):
                self.debug_visualizer.paf_raw(fields[paf_i], stride, reg_components=3)

        # confidence scales
        if self.confidence_scales:
            for paf_i in self.paf_indeces:
                paf = fields[paf_i]
                cs = np.array(self.confidence_scales, dtype=np.float32).reshape((-1, 1, 1,))
                paf[0] = cs * paf[0]

        # normalize
        normalized_pifs = [normalize_pif(*fields[pif_i], fixed_scale=self.pif_fixed_scale)
                           for pif_i in self.pif_indeces]
        normalized_pafs = [normalize_paf(*fields[paf_i], fixed_b=self.fixed_b)
                           for paf_i in self.paf_indeces]

        # pif hr
        pifhr = PifHr(self.pif_nn)
        for stride, pif, min_scale in zip(self.strides,
                                          normalized_pifs,
                                          self.pif_min_scales):
            pifhr.fill(pif, stride, min_scale=min_scale)

        # seeds
        seeds = PifSeeds(pifhr.target_accumulator, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer)
        for stride, pif, min_scale in zip(self.strides,
                                          normalized_pifs,
                                          self.pif_min_scales):
            seeds.fill(pif, stride, min_scale=min_scale)

        # paf_scored
        paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=self.paf_th)
        for stride, paf, min_distance in zip(self.strides,
                                             normalized_pafs,
                                             self.paf_min_distances):
            paf_scored.fill(paf, stride, min_distance=min_distance)

        gen = generator.Dijkstra(
            pifhr, paf_scored, seeds,
            seed_threshold=self.seed_threshold,
            connection_method=self.connection_method,
            paf_nn=self.paf_nn,
            paf_th=self.paf_th,
            skeleton=self.skeleton,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        if self.force_complete:
            gen.paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=0.0001)
            for stride, paf in zip(self.strides, normalized_pafs):
                gen.paf_scored.fill(paf, stride)
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
