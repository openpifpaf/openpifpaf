from abc import ABCMeta, abstractmethod


class Preprocess(metaclass=ABCMeta):
    """Preprocess an image with annotations and meta information."""
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""
