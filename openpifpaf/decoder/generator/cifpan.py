from collections import defaultdict
import logging
from queue import PriorityQueue
import time

import numpy as np

from .generator import Generator
from ...annotation import Annotation
from ..field_config import FieldConfig
from ..cif_hr import CifHr
from ..cif_seeds import CifSeeds
from ..caf_scored import CafScored
from .. import nms as nms_module
from ..occupancy import Occupancy
from ... import visualizer

import torch

# pylint: disable=import-error
from ...functional import caf_center_s

LOG = logging.getLogger(__name__)


class CifPan(Generator):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.0

    ball = False
    cent = True


    def __init__(self, field_config: FieldConfig, *,
                keypoints,
                #  skeleton,
                 out_skeleton=None,
                 confidence_scales=None,
                 worker_pool=None,
                 nms=True
                ):
        super().__init__(worker_pool)
        if nms is True:
            nms = nms_module.Keypoints()

        self.field_config = field_config

        self.keypoints = keypoints
        # self.skeleton = skeleton
        # self.skeleton_m1 = np.asarray(skeleton) - 1
        self.out_skeleton = out_skeleton
        self.confidence_scales = confidence_scales
        self.nms = nms

        self.timers = defaultdict(float)

        # init by_target and by_source
        # self.by_target = defaultdict(dict)
        # for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
        #     self.by_target[j2][j1] = (caf_i, True)
        #     self.by_target[j1][j2] = (caf_i, False)
        # self.by_source = defaultdict(dict)
        # for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
        #     self.by_source[j1][j2] = (caf_i, True)
        #     self.by_source[j2][j1] = (caf_i, False)

    def __call__(self, fields, initial_annotations=None):
        cif, pan = fields
        semantic, offsets = pan['semantic'], pan['offset']

        Ci, Bi = (17, object()) if self.ball else (17, 18)

        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        # print(self.field_config)
        # if self.field_config.cif_visualizers:
        #     for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
        #         vis.predicted(fields[cif_i])
        # if self.field_config.caf_visualizers:
        #     for vis, caf_i in zip(self.field_config.caf_visualizers, self.field_config.caf_indices):
        #         vis.predicted(fields[caf_i])

        print(self.field_config)
        print('pif fields',fields[0].shape)
        cifhr = CifHr(self.field_config).fill(fields)

        # seeds = CifSeeds(cifhr.accumulated, self.field_config).fill(fields)

        # caf_scored = CafScored(cifhr.accumulated, self.field_config, self.skeleton).fill(fields)

        Ñ = None

        def cif_local_max(cif):
            """Use torch for max pooling"""
            cif = torch.tensor(cif)
            cif_m = torch.max_pool2d(cif[None], 7, stride=1, padding=3)[0] == cif
            cif_m &= cif > 0.1
            return np.asarray(cif_m)

        # Get coordinates of keypoints of every type
        # list[K,N_k]
        keypoints_yx = [np.stack(np.nonzero(cif_local_max(cif)), axis=-1)
                        for cif in cifhr.accumulated]

        if len(keypoints_yx[Ci]) == 0:
            return []


        # Get instance mapping for every pixel
        # keypoints[Ci] tensor[I,2]
        # offsets       tensor[2,H,W]
        # meshgrid      tensor[2,H,W]
        # absolute = offsets + np.stack(np.meshgrid(np.arange(offsets.shape[2]),
        #                                           np.arange(offsets.shape[1])))

        absolute = offsets + np.stack(np.meshgrid(np.arange(offsets.shape[1]),
                                                  np.arange(offsets.shape[2]), indexing='ij'))
        # plt.imshow(offsets[0])
        # plt.colorbar()
        # plt.show()
        # plt.imshow(offsets[1])
        # plt.colorbar()
        # plt.show()

        difference = (absolute[Ñ,:,:,:] -                   # [ ,2,H,W]
                      keypoints_yx[Ci][:,:,Ñ,Ñ]             # [I,2, , ]
                      )

        distances2 = np.square(difference).sum(axis=1)      # [I,H,W]
        instances = distances2.argmin(axis=0)               # [H,W]

        # For each detected keypoints, get its confidence and instance
        centers_fyxv = [
            # (Ci, y, x, cifhr.accumulated[Ci,y,x])
            (Ci, y, x, 2.)
            for y, x in keypoints_yx[Ci]
        ]
        if self.ball:
            ball_fyxv = [
                (Bi, y, x, cifhr.accumulated[Bi,y,x])
                for y, x in keypoints_yx[Bi]
            ]
        keypoints_fyxiv = [
            # (f, y, x, instances[y,x], cifhr.accumulated[f,y,x])
            (f, y, x, instances[y,x], 2.)
            for f, kp_yx in enumerate(keypoints_yx[:Ci])
            for y, x in kp_yx
        ]

        annotations = []
        for f, y, x, v in centers_fyxv:
            # print('vvvv',v)
            # v = 2.
            annotation = Annotation(
                self.keypoints, self.out_skeleton,
                category_id={17:1,18:37}[f]  # center => person, ball center => ball
                )
            annotation.add(f, (x,y,v))
            annotations.append(annotation)

        # Assign keypoints to their instance (least confidence first)
        keypoints_fyxiv.sort(key=lambda x:x[-1])
        for f,y,x,i,v in keypoints_fyxiv:
            annotation = annotations[i]
            annotation.add(f, (x,y,v))

        # semantic      shape [C,H,W]
        classes = semantic.argmax(axis=0)   # [H,W]
        from matplotlib import pyplot as plt
        plt.imshow(classes)
        plt.show()
        print('show')
        # plt.savefig('data-mscoco/test.png')

        panoptic = classes*1000 + instances
        for i in range(len(annotations)):
            annotation = annotations[i]
            centroid_mask = (classes != 0) & (instances == i)
            # print(semantic.shape)
            annotation.cls = 1# semantic[:,centroid_mask].sum(axis=1).argmax(axis=0)
            annotation.mask = centroid_mask
        
        if self.ball:
            for f, y, x, v in ball_fyxv:
                annotation = Annotation().add(f, (x, y, v))
                annotation.cls = 37# semantic[:,centroid_mask].sum(axis=1).argmax(axis=0)
                annotation.mask = None
                annotations.append(annotation)

        ball_mask = classes == 2
        if ball_mask.sum() > 10:
            pass

        # self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        # if self.force_complete:
        #     annotations = self.complete_annotations(cifhr, fields, annotations)

        # if self.nms is not None:
        #     annotations = self.nms.annotations(annotations)

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations
    