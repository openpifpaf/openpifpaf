from .preprocess import Preprocess
import time

class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta):
        # print(self.preprocess_list[2])
        last_time = time.time()
        for p in self.preprocess_list:
            if p is None:
                continue
            
            image, anns, meta = p(image, anns, meta)
            print('Transform time: ', time.time() - last_time, p)
            last_time = time.time()

        # return image, anns, meta
        return image, anns, meta
