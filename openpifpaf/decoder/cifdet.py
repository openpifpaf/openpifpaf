from collections import defaultdict
import logging
import time
from typing import List

import torch
import torchvision

from .decoder import Decoder
from ..annotation import AnnotationDet
from .. import headmeta, visualizer

LOG = logging.getLogger(__name__)


class CifDet(Decoder):
    iou_threshold = 0.5
    occupancy_visualizer = visualizer.Occupancy()

    def __init__(self, head_metas: List[headmeta.CifDet], *, visualizers=None):
        super().__init__()
        self.metas = head_metas

        # prefer decoders with more classes
        self.priority = -1.0  # prefer keypoints over detections
        self.priority += sum(m.n_fields for m in head_metas) / 1000.0

        self.visualizers = visualizers
        if self.visualizers is None:
            self.visualizers = [visualizer.CifDet(meta) for meta in self.metas]

        self.cpp_decoder = torch.classes.openpifpaf_decoder.CifDet()

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

        if self.visualizers:
            for vis, meta in zip(self.visualizers, self.metas):
                vis.predicted(fields[meta.head_index])

        categories, scores, boxes = self.cpp_decoder.call(
            fields[self.metas[0].head_index],
            self.metas[0].stride,
        )
        mask = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        LOG.debug('cpp annotations = %d (%.1fms)',
                  len(mask),
                  (time.perf_counter() - start) * 1000.0)

        annotations_py = []
        boxes_np = boxes.numpy()
        for i in mask:
            ann = AnnotationDet(self.metas[0].categories)
            box = boxes_np[i]
            box[2:] -= box[:2]  # convert to xywh
            ann.set(int(categories[i]), float(scores[i]), box)
            annotations_py.append(ann)

        LOG.info('annotations %d, decoder = %.1fms',
                 len(annotations_py),
                 (time.perf_counter() - start) * 1000.0)
        return annotations_py
