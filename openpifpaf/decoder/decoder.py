from abc import ABCMeta, abstractmethod, abstractstaticmethod


class Decoder(metaclass=ABCMeta):
    @abstractstaticmethod
    def match(head_names):  # pylint: disable=unused-argument
        return False

    @classmethod
    def cli(cls, parser):
        """Add decoder specific command line arguments to the parser."""

    @classmethod
    def apply_args(cls, args):
        """Read command line arguments args to set class properties."""

    @abstractmethod
    def __call__(self, fields):
        pass
