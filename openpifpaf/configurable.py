import argparse


class Configurable:
    def __init__(self, **kwargs):
        # use kwargs to set instance attributes to overwrite class attributes
        for key, value in kwargs.items():
            assert hasattr(self, key), key
            setattr(self, key, value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    def configure(cls, args: argparse.Namespace):
        pass
