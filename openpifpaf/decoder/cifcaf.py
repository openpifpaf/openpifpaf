"""Decoder for pif-paf fields."""

import logging
import time

from . import generator
from .field_config import FieldConfig
from .paf_scored import PafScored
from .pif_hr import PifHr
from .pif_seeds import PifSeeds

LOG = logging.getLogger(__name__)


class CifCaf(object):
    force_complete = True
    connection_method = 'blend'
    paf_th = 0.1

    def __init__(self, config: FieldConfig, *,
                 keypoints,
                 skeleton,
                 out_skeleton=None,
                 debug_visualizer=None):
        self.config = config
        self.config.verify()

        self.keypoints = keypoints
        self.skeleton = skeleton
        self.out_skeleton = out_skeleton
        self.debug_visualizer = debug_visualizer

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:
            for stride, pif_i in zip(self.config.cif_strides, self.config.cif_indices):
                self.debug_visualizer.pif_raw(fields[pif_i], stride)
            for stride, paf_i in zip(self.config.caf_strides, self.config.caf_indices):
                self.debug_visualizer.paf_raw(fields[paf_i], stride)

        pifhr = PifHr(self.config).fill(fields)
        seeds = PifSeeds(pifhr.accumulated, self.config).fill(fields)
        paf_scored = PafScored(pifhr.accumulated, self.config, self.skeleton,
                               score_th=self.paf_th).fill(fields)

        gen = generator.Frontier(
            pifhr, paf_scored, seeds,
            connection_method=self.connection_method,
            keypoints=self.keypoints,
            skeleton=self.skeleton,
            out_skeleton=self.out_skeleton,
            confidence_scales=self.config.confidence_scales,
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations)
        if self.force_complete:
            gen.paf_scored = PafScored(pifhr.accumulated, self.config, self.skeleton,
                                       score_th=0.0001).fill(fields)
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
