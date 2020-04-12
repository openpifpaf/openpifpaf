from collections import defaultdict
import logging
import time

from ...annotation import AnnotationDet
from ..field_config import FieldConfig
from ..cif_hr import CifDetHr
from ..cif_seeds import CifDetSeeds

LOG = logging.getLogger(__name__)


class CifDet:
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

        annotations = [
            AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
            for v, f, x, y, w, h in seeds.get()
        ]

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
