from abc import abstractmethod
import argparse
import logging
import multiprocessing
import sys
import time
from typing import List

import torch

from .. import annotation, visualizer

LOG = logging.getLogger(__name__)


class DummyPool():
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


class Decoder:
    """Generate predictions from image or field inputs.

    When creating a new generator, the main implementation goes into `__call__()`.
    """
    default_worker_pool = None

    def __init__(self):
        self.worker_pool = self.default_worker_pool

        if self.worker_pool is None or self.worker_pool == 0:
            self.worker_pool = DummyPool()
        if isinstance(self.worker_pool, int):
            LOG.info('creating decoder worker pool with %d workers', self.worker_pool)
            assert not sys.platform.startswith('win'), (
                'not supported, use --decoder-workers=0 '
                'on windows'
            )

            # The new default for multiprocessing is 'spawn' for py38 on Mac.
            # This is not compatible with our configuration system.
            # For now, try to use 'fork'.
            # TODO: how to make configuration 'spawn' compatible
            multiprocessing_context = multiprocessing.get_context('fork')
            self.worker_pool = multiprocessing_context.Pool(self.worker_pool)

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    @classmethod
    def factory(cls, head_metas) -> List['Generator']:
        """Create instances of an implementation."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, fields, *, initial_annotations=None) -> List[annotation.Base]:
        """For single image, from fields to annotations."""
        raise NotImplementedError

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

    def batch(self, model, image_batch, *, device=None, gt_anns_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device)
        self.last_nn_time = time.perf_counter() - start_nn

        if gt_anns_batch is None:
            gt_anns_batch = [None for _ in fields_batch]

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]
            gt_anns_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch, gt_anns_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.3fs, dec = %.3fs', self.last_nn_time, self.last_decoder_time)
        return result

    def _mappable_annotations(self, fields, debug_image, gt_anns):
        if debug_image is not None:
            visualizer.Base.processed_image(debug_image)
        if gt_anns is not None:
            visualizer.Base.ground_truth(gt_anns)

        return self(fields)
