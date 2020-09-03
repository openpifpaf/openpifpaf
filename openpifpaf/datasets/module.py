import argparse
from typing import List

from .. import headmeta


class DataModule():
    """Interface for custom data.

    Subclass this class to incorporate your custom dataset.

    Overwrite cli() and configure() to make your data module configureable
    from the command line.

    Set `head_metas` in constructor.
    """
    batch_size = 8
    loader_workers = 0

    # make instance(!) variable (not class variable) in derived classes
    head_metas: List[headmeta.Base] = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    def configure(cls, args: argparse.Namespace):
        pass

    def train_loader(self):
        raise NotImplementedError

    def val_loader(self):
        raise NotImplementedError

    def test_loader(self):
        raise NotImplementedError
