from typing import List

from .preprocess import Preprocess


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        for p in self.preprocess_list:
            if p is None:
                continue
            args = p(*args)

        return args
