"""The Processor runs the model to obtain fields and passes them to a decoder."""

import logging
import multiprocessing
import time

import numpy as np
import torch

from .. import visualizer

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Processor(object):
    def __init__(self, model, decode, *,
                 device=None,
                 worker_pool=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            worker_pool = multiprocessing.Pool(worker_pool)

        self.model = model
        self.decode = decode
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

            cif_head, caf_head = self.model(image_batch)

            # to numpy
            cif_head = cif_head.cpu().numpy()
            caf_head = caf_head.cpu().numpy()

        # index by frame (item in batch)
        heads = list(zip(cif_head, caf_head))

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return heads

    def annotations_batch(self, fields_batch, *, meta_batch=None, debug_images=None):
        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
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

    def annotations(self, fields, *, initial_annotations=None, meta=None):  # pylint: disable=unused-argument
        start = time.time()

        annotations = self.decode(fields, initial_annotations=initial_annotations)

        LOG.debug('total processing time: %.3fs', time.time() - start)
        return annotations


class ProcessorDet(object):
    debug_visualizer = None

    def __init__(self, model, decode, *,
                 device=None,
                 worker_pool=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            worker_pool = multiprocessing.Pool(worker_pool)

        self.model = model
        self.decode = decode
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

        LOG.info('%d annotations', len(annotations))
        LOG.debug('total processing time: %.3fs', time.time() - start)
        return annotations
