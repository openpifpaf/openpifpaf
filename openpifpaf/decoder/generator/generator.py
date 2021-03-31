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
    def fields_batch(model, image_batch, *, device=None, target_batch=None):
        """From image batch to field batch."""
        start = time.time()

        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            if isinstance(items, dict):
                return {k: apply(f, v) for k, v in items.items()}
            return f(items)

        with torch.no_grad():
            if device is not None:
                image_batch = image_batch.to(device, non_blocking=True)

            with torch.autograd.profiler.record_function('model'):
                heads = model(image_batch)

            # to numpy
            with torch.autograd.profiler.record_function('tonumpy'):
                heads = apply(lambda x: x.cpu().numpy(), heads)
        # print('len heads', len(heads))
        # print('len heads', heads[0].shape)
        # print('len heads', heads[1].keys())
        # print('len heads', heads[2].shape)

        # print('len target_batch', len(target_batch))
        # print('len target_batch', len(target_batch[0]))
        # print('len target_batch', target_batch[0][0].shape)
        # print('len target_batch', target_batch[0][1].shape)
        # print('len target_batch', target_batch[0][2].shape)
        # print('len heads', heads[1]['semantic'].shape)
        # print('len heads', heads[1]['offset'].shape)
        # print('len target_batch', target_batch[1]['semantic'].shape)
        # print('len target_batch', target_batch[1]['offset'].shape)
        # print('heads', type(heads[1]['semantic']))
        # print('heads', type(target_batch[1]['semantic']))
        if target_batch is not None:
            import numpy as np
            def classes_from_target(semantic_target):
                ''' converting [B,H,W] to [B,C,H,W] '''
                B, H, W = semantic_target.shape
                C = semantic_target.max() + 1
                target = np.zeros((B, C, H, W))
                for cc in range(C):
                    target[:,cc,:,:] = np.where(semantic_target == cc, 1, 0)
                return target
            heads[1]['semantic'] = classes_from_target(target_batch[1]['semantic'].numpy())
            heads[1]['offset'] = target_batch[1]['offset'].numpy()
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

    def batch(self, model, image_batch, *, device=None, target_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device, target_batch=target_batch)
        # print('fields batch',len(fields_batch[0]))
        # print('fields batch',fields_batch)
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
