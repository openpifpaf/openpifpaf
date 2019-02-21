import numpy as np
import torch


class Skeleton(object):
    def __init__(self, ann_rescale, max_instances=100, n_keypoints=17):
        self.ann_rescale = ann_rescale
        self.max_instances = max_instances
        self.n_keypoints = n_keypoints

    def __call__(self, anns, width_height_original, v_th=0):
        keypoint_sets, __, ___ = self.ann_rescale(anns, width_height_original)

        out = np.zeros((self.max_instances, self.n_keypoints, 3), dtype=np.float)
        for i, keypoints in enumerate(keypoint_sets):
            for f, xyv in enumerate(keypoints):
                if xyv[2] <= v_th:
                    continue
                out[i, f] = xyv

        return (torch.from_numpy(out),)
