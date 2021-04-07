from .preprocess import Preprocess


class Encoders(Preprocess):
    """Preprocess operation that runs encoders."""
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return image, anns, meta
