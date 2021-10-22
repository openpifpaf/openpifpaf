from .preprocess import Preprocess
import time

class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta):

        for i,p in enumerate(self.preprocess_list):
            if p is None:
                continue
            
            image, anns, meta = p(image, anns, meta)
            
        return image, anns, meta
