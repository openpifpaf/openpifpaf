from collections import defaultdict
import logging
import time

from .generator import Generator
from ...annotation import AnnotationCent
from ..field_config import FieldConfig
from ..cif_hr import CifHr
from ..cif_seeds import CifSeeds
from .. import nms
from ..occupancy import Occupancy
from ... import visualizer
import numpy as np
LOG = logging.getLogger(__name__)


class CifCent(Generator):
    occupancy_visualizer = visualizer.Occupancy()

    def __init__(self, field_config: FieldConfig, keypoints, *, worker_pool=None):
        super().__init__(worker_pool)
        self.field_config = field_config
        self.keypoints = keypoints

        self.timers = defaultdict(float)

    def __call__(self, fields):
        start = time.perf_counter()

        if self.field_config.cif_visualizers:
            for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
                vis.predicted(fields[cif_i])

        cifhr = CifHr(self.field_config).fill(fields)
        ball_from_mask = [np.unravel_index(np.argmax(cifhr.accumulated[0]), cifhr.accumulated[0].shape)]
        ball_fyxv = [
            (0, y, x, cifhr.accumulated[0][y,x])
            for y, x in ball_from_mask
            ]

        
        # seeds = CifSeeds(cifhr.accumulated, self.field_config).fill(fields)
        # occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=2.0)

        annotations = []
        # def mark_occupied(ann):
        #     for joint_i, xyv in enumerate(ann.data):
        #         if xyv[2] == 0.0:
        #             continue

        #         width = ann.joint_scales[joint_i]
        #         occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma


        # for v, f, x, y, s in seeds.get():
        #     if occupied.get(f, x, y):
        #         continue

        #     ann = AnnotationCent(self.keypoints).add(f, (x, y, v))
        #     ann.joint_scales[f] = s
        #     # self._grow(ann, caf_scored)
        #     annotations.append(ann)
        #     mark_occupied(ann)

        # # for v, f, x, y, w, h in seeds.get():
        # #     if occupied.get(f, x, y):
        # #         continue
        # #     ann = AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
        # #     annotations.append(ann)
        # #     occupied.set(f, x, y, 0.1 * min(w, h))

        # self.occupancy_visualizer.predicted(occupied)

        # annotations = nms.Detection().annotations(annotations)
        # annotations = sorted(annotations, key=lambda a: -a.score)

        LOG.info('annotations %d, decoder = %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
