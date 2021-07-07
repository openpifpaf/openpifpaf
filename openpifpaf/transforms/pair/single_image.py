import random

import numpy as np
import torch

from ..preprocess import Preprocess


class SingleImage(Preprocess):
    def __init__(self, single_image_op):
        self.single_image_op = single_image_op

    def __call__(self, image_group, anns_group, meta_group):
        out_images = []
        out_anns = []
        out_meta = []

        # force the random generators to change
        _ = torch.randint(0, 10, (1,)).item()
        _ = random.random()
        _ = np.random.rand(1)

        for image, anns, meta in zip(image_group, anns_group, meta_group):
            py_rnd_state = random.getstate()
            np_rnd_state = np.random.get_state()
            with torch.random.fork_rng(devices=[]):
                image, anns, meta = self.single_image_op(image, anns, meta)
                out_images.append(image)
                out_anns.append(anns)
                out_meta.append(meta)
            random.setstate(py_rnd_state)
            np.random.set_state(np_rnd_state)

        return out_images, out_anns, out_meta


class Ungroup(Preprocess):
    """During evaluation, tracking datasets produce image groups of length
    one. Ungroup them so that it looks like any other single-image dataset.
    """
    def __call__(self, image_group, anns_group, meta_group):
        assert len(image_group) == 1
        assert len(anns_group) == 1
        assert len(meta_group) == 1
        return image_group[0], anns_group[0], meta_group[0]
