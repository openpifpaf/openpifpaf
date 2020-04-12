from collections import defaultdict
import logging
from queue import PriorityQueue
import time

import numpy as np

from ..annotation import AnnotationDet
from ..field_config import FieldConfig
from ..cif_hr import CifDetHr
from ..cif_seeds import CifDetSeeds
from ..caf_scored import CafScored
from ..occupancy import Occupancy

# pylint: disable=import-error
from ...functional import caf_center_s

LOG = logging.getLogger(__name__)


class CifDet(object):
    debug_visualizer = None

    def __init__(self, field_config: FieldConfig, categories):
        self.field_config = field_config
        self.categories = categories

        self.timers = defaultdict(float)

    def __call__(self, fields):
        start = time.perf_counter()

        if self.field_config.cif_visualizers:
            for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
                vis.predicted(fields[cif_i])

        cifhr = CifDetHr(self.field_config).fill(fields)
        seeds = CifDetSeeds(cifhr.accumulated, self.field_config).fill(fields)

        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=4)
        annotations = []

        def mark_occupied(ann):
            width = min(ann.bbox[2], ann.bbox[3])
            occupied.set(ann.field_i, ann.bbox[0], ann.bbox[1], width)

        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue

            ann = AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
            annotations.append(ann)
            mark_occupied(ann)

        if self.debug_visualizer:
            self.debug_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        return annotations
