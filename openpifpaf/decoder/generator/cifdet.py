from collections import defaultdict
import logging
import time
from typing import List

from .generator import Generator
from ...annotation import AnnotationDet
from ..cif_hr import CifDetHr
from ..cif_seeds import CifDetSeeds
from .. import nms
from ..occupancy import Occupancy
from ... import headmeta, visualizer

LOG = logging.getLogger(__name__)


class CifDet(Generator):
    occupancy_visualizer = visualizer.Occupancy()

    def __init__(self, head_metas: List[headmeta.CifDet], *, visualizers=None):
        super().__init__()
        self.metas = head_metas

        self.visualizers = visualizers
        if self.visualizers is None:
            self.visualizers = [visualizer.CifDet(meta) for meta in self.metas]

        self.timers = defaultdict(float)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            CifDet([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.CifDet)
        ]

    def __call__(self, fields):
        start = time.perf_counter()
        cifdet_fields = [fields[meta.head_index] for meta in self.metas]

        if self.visualizers:
            for vis, meta in zip(self.visualizers, self.metas):
                vis.predicted(fields[meta.head_index])

        cifhr = CifDetHr().fill(cifdet_fields, self.metas)
        seeds = CifDetSeeds(cifhr.accumulated).fill(cifdet_fields, self.metas)
        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=2.0)

        annotations = []
        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.metas[0].categories).set(
                f + 1, v, (x - w/2.0, y - h/2.0, w, h))
            annotations.append(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        self.occupancy_visualizer.predicted(occupied)

        annotations = nms.Detection().annotations(annotations)
        # annotations = sorted(annotations, key=lambda a: -a.score)

        LOG.info('annotations %d, decoder = %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
