from collections import defaultdict
import logging
import time

import torch

from .generator import Generator
from ...annotation import AnnotationDet
from ..field_config import FieldConfig
from ..cif_hr import CifDetHr
from ..cif_seeds import CifDetSeeds
from .. import nms
from ..occupancy import Occupancy

LOG = logging.getLogger(__name__)


class CifDet(Generator):
    def __init__(self, field_config: FieldConfig, categories, *, worker_pool=None):
        super().__init__(worker_pool)
        self.field_config = field_config
        self.categories = categories

        self.timers = defaultdict(float)

    @staticmethod
    def fields_batch(model, image_batch, *, device=None):
        start = time.time()
        with torch.no_grad():
            if device is not None:
                image_batch = image_batch.to(device, non_blocking=True)

            cif_head, _ = model(image_batch)

            # to numpy
            cif_head = cif_head.cpu().numpy()

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return [(ch,) for ch in cif_head]

    def __call__(self, fields):
        start = time.perf_counter()

        if self.field_config.cif_visualizers:
            for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
                vis.predicted(fields[cif_i])

        cifhr = CifDetHr(self.field_config).fill(fields)
        seeds = CifDetSeeds(cifhr.accumulated, self.field_config).fill(fields)
        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=2.0)

        annotations = []
        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
            annotations.append(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        annotations = nms.Detection().annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
