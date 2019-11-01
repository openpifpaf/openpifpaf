"""The Processor runs the model to obtain fields and passes them to a decoder."""

import cProfile
import io
import logging
import multiprocessing
import pstats
import time

import numpy as np
import torch

from .utils import scalar_square_add_single

# pylint: disable=import-error
from ..functional import scalar_nonzero_clipped

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Processor(object):
    def __init__(self, model, decode, *,
                 output_stride=None,
                 keypoint_threshold=0.0, instance_threshold=0.0,
                 debug_visualizer=None,
                 profile=None,
                 device=None,
                 worker_pool=None,
                 suppressed_v=0.0,
                 instance_scorer=None):
        if profile is True:
            profile = cProfile.Profile()

        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            worker_pool = multiprocessing.Pool(worker_pool)

        self.model = model
        self.decode = decode
        self.output_stride = output_stride or model.head_strides[-1]
        self.keypoint_threshold = keypoint_threshold
        self.instance_threshold = instance_threshold
        self.debug_visualizer = debug_visualizer
        self.profile = profile
        self.device = device
        self.worker_pool = worker_pool
        self.suppressed_v = suppressed_v
        self.instance_scorer = instance_scorer

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('model', 'worker_pool', 'device')
        }

    def set_cpu_image(self, cpu_image, processed_image):
        if self.debug_visualizer is not None:
            self.debug_visualizer.set_image(cpu_image, processed_image)

    def fields(self, image_batch):
        start = time.time()
        if self.device is not None:
            image_batch = image_batch.to(self.device, non_blocking=True)

        with torch.no_grad():
            heads = self.model(image_batch)

            # to numpy
            fields = [[field.cpu().numpy() for field in head] for head in heads]

            # index by batch entry
            fields = [
                [[field[i] for field in head] for head in fields]
                for i in range(image_batch.shape[0])
            ]

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return fields

    def soft_nms(self, annotations):
        if not annotations:
            return annotations

        occupied = np.zeros((
            len(annotations[0].data),
            int(max(np.max(ann.data[:, 1]) for ann in annotations) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in annotations) + 1),
        ), dtype=np.uint8)

        annotations = sorted(annotations, key=lambda a: -a.score())
        for ann in annotations:
            joint_scales = (np.maximum(4.0, ann.joint_scales)
                            if ann.joint_scales is not None
                            else np.ones((ann.data.shape[0]),) * 4.0)

            assert len(occupied) == len(ann.data)
            for xyv, occ, joint_s in zip(ann.data, occupied, joint_scales):
                v = xyv[2]
                if v == 0.0:
                    continue

                if scalar_nonzero_clipped(occ, xyv[0], xyv[1]):
                    xyv[2] = self.suppressed_v
                else:
                    scalar_square_add_single(occ, xyv[0], xyv[1], joint_s, 1)

        if self.debug_visualizer is not None:
            LOG.debug('Occupied fields after NMS')
            self.debug_visualizer.occupied(occupied[0])
            self.debug_visualizer.occupied(occupied[4])

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
        return self.annotations(fields, meta=meta, debug_image=debug_image)

    def suppress_outside_valid(self, ann, valid_area):
        m = np.logical_or(
            np.logical_or(ann.data[:, 0] < valid_area[0],
                          ann.data[:, 0] > valid_area[0] + valid_area[2]),
            np.logical_or(ann.data[:, 1] < valid_area[1],
                          ann.data[:, 1] > valid_area[1] + valid_area[3]),
        )
        ann.data[m, 2] = np.maximum(ann.data[m, 2], self.suppressed_v)

    def annotations(self, fields, *, initial_annotations=None, meta=None, debug_image=None):
        start = time.time()
        if self.profile is not None:
            self.profile.enable()

        if debug_image is not None:
            self.set_cpu_image(None, debug_image)

        annotations = self.decode(fields, initial_annotations=initial_annotations)

        # instance scorer
        if self.instance_scorer is not None:
            for ann in annotations:
                ann.fixed_score = self.instance_scorer.from_annotation(ann)

        # nms
        annotations = self.soft_nms(annotations)

        # threshold
        for ann in annotations:
            if meta is not None:
                self.suppress_outside_valid(ann, meta['valid_area'])
            kps = ann.data
            kps[kps[:, 2] < self.keypoint_threshold] = 0.0
        annotations = [ann for ann in annotations
                       if ann.score() >= self.instance_threshold]
        annotations = sorted(annotations, key=lambda a: -a.score())

        if self.profile is not None:
            self.profile.disable()
            iostream = io.StringIO()
            ps = pstats.Stats(self.profile, stream=iostream)
            ps = ps.sort_stats('tottime')
            ps.print_stats()
            ps.dump_stats('decoder.prof')
            print(iostream.getvalue())

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        LOG.debug('total processing time: %.3fs', time.time() - start)
        return annotations
