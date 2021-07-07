import copy
import numpy as np

from .base import Base
from .caf import Caf
from .. import headmeta


class Tcaf(Base):
    def __init__(self, meta: headmeta.Tcaf):
        super().__init__(meta.name)
        self.meta = meta
        self.caf_visualizer = Caf(meta)

    @staticmethod
    def merge_anns(ann1, ann2):
        m = copy.deepcopy(ann1)
        m['keypoints'] = np.concatenate((ann1['keypoints'], ann2['keypoints']), axis=0)
        return m

    def targets(self, field, *, annotation_dicts):
        anns1, anns2 = annotation_dicts

        anns1_by_trackid = {ann['track_id']: ann for ann in anns1}
        merged_annotations = [
            self.merge_anns(anns1_by_trackid[ann2['track_id']], ann2)
            for ann2 in anns2
            if (not ann2['iscrowd']
                and ann2['track_id'] in anns1_by_trackid)
        ]

        self.caf_visualizer.targets(field, annotation_dicts=merged_annotations)

    def predicted(self, field):
        self.caf_visualizer.predicted(field)
