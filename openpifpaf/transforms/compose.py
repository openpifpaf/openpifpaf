from .preprocess import Preprocess


class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, mask, meta):
        # print(self.preprocess_list[2])
        for p in self.preprocess_list:
            if p is None:
                continue
            # image, anns, meta = p(image, anns, meta)
            ### AMA
            # print('++++++++++++++++++++++++++++++++++++')
            image, anns, mask, meta = p(image, anns, mask, meta)

        # return image, anns, meta
        return image, anns, mask, meta
