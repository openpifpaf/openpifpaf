from abc import abstractmethod
import logging
import multiprocessing
import sys
import time

import torch

from ... import visualizer

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Generator:
    def __init__(self, worker_pool=None):
        if worker_pool is None or worker_pool == 0:
            worker_pool = DummyPool()
        if isinstance(worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', worker_pool)
            assert not sys.platform.startswith('win'), (
                'not supported, use --decoder-workers=0 '
                'on windows'
            )
            worker_pool = multiprocessing.Pool(worker_pool)

        self.worker_pool = worker_pool

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ('worker_pool',)
        }

    @staticmethod
    def fields_batch(model, image_batch, *, device=None):
        """From image batch to field batch."""
        start = time.time()

        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            return f(items)

        with torch.no_grad():
            if device is not None:
                image_batch = image_batch.to(device, non_blocking=True)

            with torch.autograd.profiler.record_function('model'):
                heads = model(image_batch)

            # to numpy
            with torch.autograd.profiler.record_function('tonumpy'):
                heads = apply(lambda x: x.cpu().numpy(), heads)

        # index by frame (item in batch)
        head_iter = apply(iter, heads)
        heads = []
        while True:
            try:
                heads.append(apply(next, head_iter))
            except StopIteration:
                break

        LOG.debug('nn processing time: %.3fs', time.time() - start)
        return heads

    @abstractmethod
    def __call__(self, fields, *, initial_annotations=None):
        """For single image, from fields to annotations."""
        raise NotImplementedError()

    def batch(self, model, image_batch, *, device=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device)
        self.last_nn_time = time.perf_counter() - start_nn

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.3fs, dec = %.3fs', self.last_nn_time, self.last_decoder_time)
        return result

    def _mappable_annotations(self, fields, debug_image):
        if debug_image is not None:
            visualizer.BaseVisualizer.processed_image(debug_image)

        return self(fields)
