from .preprocess import Preprocess


class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta):
        for p in self.preprocess_list:
            if p is None:
                continue
            image, anns, meta = p(image, anns, meta)

        return image, anns, meta
