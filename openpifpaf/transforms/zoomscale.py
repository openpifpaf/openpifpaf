
import PIL
import copy

from .preprocess import Preprocess

class ZoomScale(Preprocess):
    def __init__(self):
        self.zoom = 1

    def __call__(self, image, anns, meta):
        w, h = image.size
        target_w, target_h = (w/self.zoom, h/self.zoom)
        
        top = int((h - target_h)/2)
        bottom = int(target_h + (h - target_h)/2)
        left = int((w - target_w)/2)
        right = int(target_w + (w - target_w)/2)

        # change annotations
        anns = copy.deepcopy(anns)
        for ann in anns:
            ann['bmask'] = ann['bmask'][top:bottom, left:right]
            
        return image, anns, meta