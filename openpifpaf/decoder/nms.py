import logging
import time

import numpy as np

from .occupancy import Occupancy

LOG = logging.getLogger(__name__)


class Keypoints:
    suppression = 0.0
    instance_threshold = 0.0
    keypoint_threshold = 0.0
    occupancy_visualizer = None

    def annotations(self, anns):
        start = time.perf_counter()

        for ann in anns:
            ann.data[ann.data[:, 2] < self.keypoint_threshold] = 0.0
        anns = [ann for ann in anns if ann.score() >= self.instance_threshold]

        if not anns:
            return anns

        occupied = Occupancy((
            len(anns[0].data),
            int(max(np.max(ann.data[:, 1]) for ann in anns) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in anns) + 1),
        ), 2, min_scale=4)

        anns = sorted(anns, key=lambda a: -a.score())
        for ann in anns:
            assert ann.joint_scales is not None
            assert len(occupied) == len(ann.data)
            for f, (xyv, joint_s) in enumerate(zip(ann.data, ann.joint_scales)):
                v = xyv[2]
                if v == 0.0:
                    continue

                if occupied.get(f, xyv[0], xyv[1]):
                    xyv[2] *= self.suppression
                else:
                    occupied.set(f, xyv[0], xyv[1], joint_s)  # joint_s = 2 * sigma

        if self.occupancy_visualizer is not None:
            LOG.debug('Occupied fields after NMS')
            self.occupancy_visualizer.predicted(occupied)

        for ann in anns:
            ann.data[ann.data[:, 2] < self.keypoint_threshold] = 0.0
        anns = [ann for ann in anns if ann.score() >= self.instance_threshold]
        anns = sorted(anns, key=lambda a: -a.score())

        LOG.debug('nms = %.3f', time.perf_counter() - start)
        return anns


class Detection:
    suppression = 0.1
    suppression_soft = 0.3
    instance_threshold = 0.1
    iou_threshold = 0.7
    iou_threshold_soft = 0.5

    @staticmethod
    def bbox_iou(box, other_boxes):
        box = np.expand_dims(box, 0)
        x1 = np.maximum(box[:, 0], other_boxes[:, 0])
        y1 = np.maximum(box[:, 1], other_boxes[:, 1])
        x2 = np.minimum(box[:, 0] + box[:, 2], other_boxes[:, 0] + other_boxes[:, 2])
        y2 = np.minimum(box[:, 1] + box[:, 3], other_boxes[:, 1] + other_boxes[:, 3])
        inter_area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        box_area = box[:, 2] * box[:, 3]
        other_areas = other_boxes[:, 2] * other_boxes[:, 3]
        return inter_area / (box_area + other_areas - inter_area + 1e-5)

    def annotations(self, anns):
        start = time.perf_counter()

        anns = [ann for ann in anns if ann.score >= self.instance_threshold]
        if not anns:
            return anns
        anns = sorted(anns, key=lambda a: -a.score)

        all_boxes = np.stack([ann.bbox for ann in anns])
        for ann_i, ann in enumerate(anns[1:], start=1):
            mask = [ann.score >= self.instance_threshold for ann in anns[:ann_i]]
            ious = self.bbox_iou(ann.bbox, all_boxes[:ann_i][mask])
            max_iou = np.max(ious)

            if max_iou > self.iou_threshold:
                ann.score *= self.suppression
            elif max_iou > self.iou_threshold_soft:
                ann.score *= self.suppression_soft

        anns = [ann for ann in anns if ann.score >= self.instance_threshold]
        anns = sorted(anns, key=lambda a: -a.score)

        LOG.debug('nms = %.3f', time.perf_counter() - start)
        return anns
