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
                 head_indices,
                 skeleton,
                 seed_threshold=0.2,
                 confidence_scales=None,
                 debug_visualizer=None):
        self.head_indices = head_indices
        self.skeleton = skeleton

        self.stride = stride
        self.seed_threshold = seed_threshold
        self.debug_visualizer = debug_visualizer

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

        self.confidence_scales = confidence_scales

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        pif, paf = fields[self.head_indices[0]], fields[self.head_indices[1]]
        if self.confidence_scales:
            cs = np.array(self.confidence_scales).reshape((-1, 1, 1))
            # print(paf[0].shape, cs.shape)
            # print('applying cs', cs)
            paf[0] = np.copy(paf[0])
            paf[0] *= cs
        if self.debug_visualizer:
            self.debug_visualizer.pif_raw(pif, self.stride)
            self.debug_visualizer.paf_raw(paf, self.stride, reg_components=3)
        paf = normalize_paf(*paf, fixed_b=self.fixed_b)
        pif = normalize_pif(*pif, fixed_scale=self.pif_fixed_scale)
        pifhr = PifHr(self.pif_nn).fill(pif, self.stride)
        seeds = PifSeeds(pifhr.target_accumulator, self.seed_threshold,
                         debug_visualizer=self.debug_visualizer).fill(pif, self.stride).get()
        paf_scored = PafScored(pifhr.targets, self.skeleton,
                               score_th=self.paf_th).fill(paf, self.stride)

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
            gen.paf_scored = PafScored(pifhr.targets, self.skeleton,
                                       score_th=0.0001).fill(paf, self.stride)
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
