import logging
from typing import List

from .module import DataModule
from .multiloader import MultiLoader

LOG = logging.getLogger(__name__)


class ConcatenatedLists:
    """Special treatment of set item: operation happens in underlying list.

    Therefore,
    c[15] = 3
    might set the fifth item in the third list to 3. Regular list concatenations
    would not modify the underlying lists.
    """
    def __init__(self, lists):
        self.lists = lists

    def __len__(self):
        return sum(len(l) for l in self.lists)

    def __getitem__(self, key):
        for l in self.lists:
            if key < len(l):
                return l[key]
            key -= len(l)
        raise KeyError

    def __setitem__(self, key, value):
        for l in self.lists:
            if key < len(l):
                l[key] = value
                return
            key -= len(l)
        raise KeyError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class MultiDataModule(DataModule):
    """Emulates a single DataModule but contains multiple DataModules."""

    def __init__(self, datamodules: List[DataModule]):
        self.datamodules = datamodules

        self.head_metas = ConcatenatedLists([dm.head_metas for dm in datamodules])

        LOG.info('%d data modules with %d head metas',
                 len(self.datamodules), len(self.head_metas))

    def metrics(self):
        return [m for dm in self.datamodules for m in dm.metrics()]

    def train_loader(self):
        return MultiLoader([dm.train_loader() for dm in self.datamodules], len(self.head_metas))

    def val_loader(self):
        return MultiLoader([dm.val_loader() for dm in self.datamodules], len(self.head_metas))

    def eval_loader(self):
        return MultiLoader([dm.eval_loader() for dm in self.datamodules], len(self.head_metas))

    def distributed_sampler(self, loader: MultiLoader):
        assert len(self.datamodules) == len(loader.loaders)
        return MultiLoader(
            [dm.distributed_sampler(l) for dm, l in zip(self.datamodules, loader.loaders)],
            len(self.head_metas),
        )
