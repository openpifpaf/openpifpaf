from collections import defaultdict
import logging

import torch

LOG = logging.getLogger(__name__)


class RunningCache(torch.nn.Module):
    def __init__(self, cached_items):
        super().__init__()

        self.cached_items = cached_items
        self.duration = abs(min(cached_items)) + 1
        self.cache = [None for _ in range(self.duration)]
        self.index = 0

        LOG.debug('running cache of length %d', len(self.cache))

    def incr(self):
        self.index = (self.index + 1) % self.duration

    def get_index(self, index):
        while index < 0:
            index += self.duration
        while index >= self.duration:
            index -= self.duration
        LOG.debug('retrieving cache at index %d', index)

        v = self.cache[index]
        if v is not None:
            v = v.detach()
        return v

    def get(self):
        return [self.get_index(i + self.index) for i in self.cached_items]

    def set_next(self, data):
        self.incr()
        self.cache[self.index] = data
        LOG.debug('set new data at index %d', self.index)
        return self

    def forward(self, *args):
        LOG.debug('----------- running cache --------------')
        x = args[0]

        o = []
        for x_i in x:
            o += self.set_next(x_i).get()

        if any(oo is None for oo in o):
            o = [oo if oo is not None else o[0] for oo in o]

        # drop images of the wrong size (determine size by majority vote)
        if len(o) >= 2:
            image_sizes = [tuple(oo.shape[-2:]) for oo in o]
            if not all(ims == image_sizes[0] for ims in image_sizes[1:]):
                freq = defaultdict(int)
                for ims in image_sizes:
                    freq[ims] += 1
                max_freq = max(freq.values())
                ref_image_size = next(iter(ims for ims, f in freq.items() if f == max_freq))

                for i, ims in enumerate(image_sizes):
                    if ims == ref_image_size:
                        continue
                    for s in range(1, len(image_sizes)):
                        target_i = (i + s) % len(image_sizes)
                        if image_sizes[target_i] == ref_image_size:
                            break
                    LOG.warning('replacing %d (%s) with %d (%s) for ref %s',
                                i, ims,
                                target_i, image_sizes[target_i],
                                ref_image_size)
                    o[i] = o[target_i]

        return torch.stack(o)
