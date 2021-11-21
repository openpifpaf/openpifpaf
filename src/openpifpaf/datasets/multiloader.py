import logging
from typing import List

import numpy as np
import torch

LOG = logging.getLogger(__name__)


class MultiSamplerProxy:
    def __init__(self, loaders: List[torch.utils.data.DataLoader]):
        self.loaders = loaders

    def __getattribute__(self, name):
        if name == 'set_epoch' and not hasattr(self.loaders[0], 'set_epoch'):
            raise AttributeError

        return object.__getattribute__(self, name)

    def set_epoch(self, value):
        for loader_i, loader in enumerate(self.loaders):
            LOG.info('setting epoch %d for loader %d', value, loader_i)
            loader.sampler.set_epoch(value)


class MultiLoader:
    last_task_index = None
    weights = None

    def __init__(self, loaders: List[torch.utils.data.DataLoader], n_heads: int, *, n_batches=None):
        self.loaders = loaders
        self.n_heads = n_heads
        self.sampler = MultiSamplerProxy(loaders)
        self._weights = self.weights

        if self._weights is None:
            self._weights = [1.0 / len(loaders) for _ in range(len(loaders))]
        elif len(self._weights) == len(loaders) - 1:
            self._weights.append(1.0 - sum(self._weights))
        elif len(self._weights) == len(loaders):
            pass
        else:
            raise Exception('invalid dataset weights: {}'.format(self._weights))
        assert all(w > 0.0 for w in self._weights)
        sum_w = sum(self._weights)
        self._weights = [w / sum_w for w in self._weights]
        LOG.info('dataset weights: %s', self._weights)

        self.n_batches = int(min(len(l) / w for l, w in zip(loaders, self._weights)))
        if n_batches:
            self.n_batches = min(self.n_batches, n_batches)

    def __iter__(self):
        loader_iters = [iter(l) for l in self.loaders]
        n_loaded = [0 for _ in self.loaders]
        while True:
            loader_index = int(np.argmin([n / w for n, w in zip(n_loaded, self._weights)]))
            next_batch = next(loader_iters[loader_index], None)
            if next_batch is None:
                break
            n_loaded[loader_index] += 1
            MultiLoader.last_task_index = loader_index

            # convert targets to multi-targets
            image_batch, target_batch, meta_batch = next_batch
            multi_target_batch = [None for _ in range(self.n_heads)]
            for i, tb in zip(meta_batch[0]['head_indices'], target_batch):
                multi_target_batch[i] = tb

            yield image_batch, multi_target_batch, meta_batch

            if sum(n_loaded) >= self.n_batches:
                break

    def __len__(self):
        return self.n_batches
