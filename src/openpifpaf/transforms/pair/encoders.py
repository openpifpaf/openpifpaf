from ..preprocess import Preprocess


class Encoders(Preprocess):
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, images, all_anns, metas):
        targets = [enc(images, all_anns, metas) for enc in self.encoders]
        meta = metas[0]
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return images, targets, meta
