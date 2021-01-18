from abc import ABCMeta, abstractmethod


class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""
