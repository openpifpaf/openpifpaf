"""The Processor runs the model to obtain fields and passes them to a decoder."""

import logging
import multiprocessing
import time

import numpy as np
import torch

from .occupancy import Occupancy
from .. import visualizer

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Processor(object):
    debug_visualizer = None

    def __init__(self, model, decode, *,
                 keypoint_threshold=0.0, instance_threshold=0.0,
                 device=None,
                 worker_pool=None,
                 suppressed_v=0.0,
                 instance_scorer=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            worker_pool = multiprocessing.Pool(worker_pool)

        self.model = model
        self.decode = decode
        self.keypoint_threshold = keypoint_threshold
        self.instance_threshold = instance_threshold
        self.device = device
        self.worker_pool = worker_pool
        self.suppressed_v = suppressed_v
        self.instance_scorer = instance_scorer

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('model', 'worker_pool', 'device')
        }

    def fields(self, image_batch):
        start = time.time()
        with torch.no_grad():
            if self.device is not None:
                image_batch = image_batch.to(self.device, non_blocking=True)

            cif_head, caf_head = self.model(image_batch)

            # to numpy
            cif_head = cif_head.cpu().numpy()
            caf_head = caf_head.cpu().numpy()

        # index by frame (item in batch)
        heads = list(zip(cif_head, caf_head))

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return heads

    def soft_nms(self, annotations):
        if not annotations:
            return annotations

        occupied = Occupancy((
            len(annotations[0].data),
            int(max(np.max(ann.data[:, 1]) for ann in annotations) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in annotations) + 1),
        ), 2, min_scale=4)

        annotations = sorted(annotations, key=lambda a: -a.score())
        for ann in annotations:
            assert ann.joint_scales is not None
            assert len(occupied) == len(ann.data)
            for f, (xyv, joint_s) in enumerate(zip(ann.data, ann.joint_scales)):
                v = xyv[2]
                if v == 0.0:
                    continue

                if occupied.get(f, xyv[0], xyv[1]):
                    xyv[2] *= self.suppressed_v
                else:
                    occupied.set(f, xyv[0], xyv[1], joint_s)  # joint_s = 2 * sigma

        if self.debug_visualizer is not None:
            LOG.debug('Occupied fields after NMS')
            self.debug_visualizer.predicted(occupied)

        annotations = [ann for ann in annotations if np.any(ann.data[:, 2] > self.suppressed_v)]
        annotations = sorted(annotations, key=lambda a: -a.score())
        return annotations

    def keypoint_sets(self, fields):
        annotations = self.annotations(fields)
        return self.keypoint_sets_from_annotations(annotations)

    @staticmethod
    def keypoint_sets_from_annotations(annotations):
        keypoint_sets = [ann.data for ann in annotations]
        scores = [ann.score() for ann in annotations]
        assert len(scores) == len(keypoint_sets)
        if not keypoint_sets:
            return np.zeros((0, 17, 3)), np.zeros((0,))
        keypoint_sets = np.array(keypoint_sets)
        scores = np.array(scores)

        return keypoint_sets, scores

    def annotations_batch(self, fields_batch, *, meta_batch=None, debug_images=None):
        if debug_images is None or self.debug_visualizer is None:
            # remove debug_images if there is no visualizer to save
            # time during pickle
            debug_images = [None for _ in fields_batch]
        if meta_batch is None:
            meta_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        return self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, meta_batch, debug_images))

    def _mappable_annotations(self, fields, meta, debug_image):
        if debug_image is not None:
            visualizer.BaseVisualizer.processed_image(debug_image)

        return self.annotations(fields, meta=meta)

    def suppress_outside_valid(self, ann, valid_area):
        m = np.logical_or(
            np.logical_or(ann.data[:, 0] < valid_area[0],
                          ann.data[:, 0] > valid_area[0] + valid_area[2]),
            np.logical_or(ann.data[:, 1] < valid_area[1],
                          ann.data[:, 1] > valid_area[1] + valid_area[3]),
        )
        ann.data[m, 2] *= self.suppressed_v

    def annotations(self, fields, *, initial_annotations=None, meta=None):
        start = time.time()

        annotations = self.decode(fields, initial_annotations=initial_annotations)

        # instance scorer
        if self.instance_scorer is not None:
            for ann in annotations:
                ann.fixed_score = self.instance_scorer.from_annotation(ann)

        # keypoint threshold
        for ann in annotations:
            if meta is not None:
                self.suppress_outside_valid(ann, meta['valid_area'])
            kps = ann.data
            kps[kps[:, 2] < self.keypoint_threshold] = 0.0

        # nms
        annotations = self.soft_nms(annotations)

        # instance threshold
        annotations = [ann for ann in annotations
                       if ann.score() >= self.instance_threshold]
        annotations = sorted(annotations, key=lambda a: -a.score())

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        LOG.debug('total processing time: %.3fs', time.time() - start)
        return annotations


class ProcessorDet(object):
    debug_visualizer = None

    def __init__(self, model, decode, *,
                 instance_threshold=0.0,
                 device=None,
                 worker_pool=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            worker_pool = multiprocessing.Pool(worker_pool)

        self.model = model
        self.decode = decode
        self.instance_threshold = instance_threshold
        self.device = device
        self.worker_pool = worker_pool

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('model', 'worker_pool', 'device')
        }

    def fields(self, image_batch):
        start = time.time()
        with torch.no_grad():
            if self.device is not None:
                image_batch = image_batch.to(self.device, non_blocking=True)

            cif_head, _ = self.model(image_batch)

            # to numpy
            cif_head = cif_head.cpu().numpy()

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return [(ch,) for ch in cif_head]

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

    def soft_nms(self, annotations, iou_th=0.7):
        if not annotations:
            return annotations

        annotations = [ann for ann in annotations if ann.score >= self.instance_threshold]
        annotations = sorted(annotations, key=lambda a: -a.score)

        all_boxes = np.stack([ann.bbox for ann in annotations])
        for ann_i, ann in enumerate(annotations[1:], start=1):
            mask = [ann.score >= self.instance_threshold for ann in annotations[:ann_i]]
            ious = self.bbox_iou(ann.bbox, all_boxes[:ann_i][mask])
            if np.max(ious) < iou_th:
                continue

            ann.score *= 0.1

        annotations = [ann for ann in annotations if ann.score >= self.instance_threshold]
        annotations = sorted(annotations, key=lambda a: -a.score)
        return annotations

    def annotations_batch(self, fields_batch, *, meta_batch=None, debug_images=None):
        if debug_images is None or self.debug_visualizer is None:
            # remove debug_images if there is no visualizer to save
            # time during pickle
            debug_images = [None for _ in fields_batch]
        if meta_batch is None:
            meta_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        return self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, meta_batch, debug_images))

    def _mappable_annotations(self, fields, meta, debug_image):
        if debug_image is not None:
            visualizer.BaseVisualizer.processed_image(debug_image)

        return self.annotations(fields, meta=meta)

    def annotations(self, fields, *, meta=None):  # pylint: disable=unused-argument
        start = time.time()

        annotations = self.decode(fields)
        annotations = self.soft_nms(annotations)

        LOG.info('%d annotations', len(annotations))
        LOG.debug('total processing time: %.3fs', time.time() - start)
        return annotations
