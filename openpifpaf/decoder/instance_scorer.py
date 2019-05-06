import copy
import json

import numpy as np
import torch


class InstanceScorer(torch.nn.Module):
    def __init__(self, in_features=35):
        super(InstanceScorer, self).__init__()
        self.compute_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.compute_layers(x - 0.5)

    def from_annotation(self, ann):
        v = torch.tensor([ann.scale()] +
                         ann.data[:, 2].tolist() +
                         ann.joint_scales.tolist()).float()
        with torch.no_grad():
            return float(self.forward(v).item())


class InstanceScoreRecorder(object):
    def __init__(self):
        """Drop in replacement for InstanceScorer that records the
        ground truth dataset instead."""
        self.data = []
        self.next_gt = None

    def set_gt(self, gt):
        assert self.next_gt is None
        self.next_gt = gt

    def from_annotation(self, annotations):
        gt = copy.deepcopy(self.next_gt)
        for ann in annotations:
            kps = ann.data

            matched = None
            for ann_gt in gt:
                kps_gt = ann_gt['keypoints']
                mask = kps_gt[:, 2] > 0
                if not np.any(mask):
                    continue

                diff = kps[mask, :2] - kps_gt[mask, :2]
                dist = np.mean(np.abs(diff))
                if dist > 10.0:
                    continue

                matched = ann_gt
                break

            if matched is None:
                self.data.append((ann, 0))
                continue

            # found a match
            self.data.append((ann, 1))
            gt.remove(matched)

        self.next_gt = None

    def write_data(self, filename):
        with open(filename, 'w') as f:
            for ann, y in self.data:
                f.write(json.dumps({
                    'keypoints': ann.data.tolist(),
                    'joint_scales': ann.joint_scales.tolist(),
                    'score': ann.score(),
                    'scale': ann.scale(),
                    'target': y,
                }))
                f.write('\n')
