from .preprocess import Preprocess


class Encoders(Preprocess):
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        return image, anns, meta
