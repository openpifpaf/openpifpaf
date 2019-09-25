"""Decoder for pif-paf fields."""

import logging
import time

from . import generator
from .decoder import Decoder
from .paf_scored import PafScored
from .pif_hr import PifHr
from .pif_seeds import PifSeeds
from .utils import normalize_pif, normalize_paf

LOG = logging.getLogger(__name__)


class PifPaf(Decoder):
    force_complete = True
    connection_method = 'max'
    fixed_b = None
    pif_fixed_scale = None
    paf_th = 0.1

    def __init__(self, stride, *,
                 skeleton,
                 pif_index=0, paf_index=1,
                 seed_threshold=0.2,
                 debug_visualizer=None):
        self.strides = stride
        self.pif_indeces = pif_index
        self.paf_indeces = paf_index
        if not isinstance(self.strides, (list, tuple)):
            self.strides = [self.strides]
            self.pif_indeces = [self.pif_indeces]
            self.paf_indeces = [self.paf_indeces]

        self.skeleton = skeleton

        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, pif_i in zip(self.strides, self.pif_indeces):
                self.debug_visualizer.pif_raw(fields[pif_i], stride)
            for stride, paf_i in zip(self.strides, self.pif_indeces):
                self.debug_visualizer.paf_raw(fields[paf_i], stride, reg_components=3)

        # normalize
        normalized_pifs = [normalize_pif(*fields[pif_i], fixed_scale=self.pif_fixed_scale)
                           for pif_i in self.pif_indeces]
        normalized_pafs = [normalize_paf(*fields[paf_i], fixed_b=self.fixed_b)
                           for paf_i in self.paf_indeces]

        # pif hr
        pifhr = PifHr(self.pif_nn)
        for stride, pif in zip(self.strides, normalized_pifs):
            pifhr.fill(pif, stride)

        # seeds
        seeds = PifSeeds(pifhr.target_accumulator, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer)
        for stride, pif in zip(self.strides, normalized_pifs):
            seeds.fill(pif, stride)

        # paf_scored
        paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=self.paf_th)
        for stride, paf in zip(self.strides, normalized_pafs):
            paf_scored.fill(paf, stride)

        gen = generator.Greedy(
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
