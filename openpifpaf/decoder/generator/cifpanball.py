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

def offsets_to_colorwheel(offset):
    import kornia, math
    offset = torch.tensor(offset)
    angle = torch.atan2(offset[:,0],offset[:,1])[:,None]
    magnitude = offset.pow(2).sum(dim=1,keepdim=True).sqrt()
    eps = 1e-3
    h = angle+math.pi
    v = torch.ones_like(angle)-eps
    s = magnitude/175
    #s[self.instance_truth("foreground_instances")==0] = 0
    v = 1-s/7
    s[s>1] = 1
    hsv = torch.cat([h,s,v], dim=1)
    rgb = kornia.color.hsv_to_rgb(hsv)
    return rgb.permute(0,2,3,1).cpu().numpy()

def softmax(semantic):
    semantic = torch.tensor(semantic)
    p = torch.softmax(semantic, dim=1)
    return p.cpu().numpy()


class CifPanBall(Generator):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.0

    ball = True
    cent = True


    def __init__(self, field_config: FieldConfig, field_config_ball: FieldConfig, *,
                keypoints,
                #  skeleton,
                field_config_cent=None,
                 out_skeleton=None,
                 confidence_scales=None,
                 worker_pool=None,
                 nms=True,
                 kp_ball=None,
                 adaptive_max_pool_th=False,
                 max_pool_th=0.1,
                 decode_masks_first=False,
                 only_output_17=False,
                 disable_pred_filter=False,
                 dec_filter_smaller_than=100,
                 dec_filter_less_than=5,
                ):
        super().__init__(worker_pool)
        if nms is True:
            nms = nms_module.Keypoints()

        self.field_config = field_config
        self.field_config_ball = field_config_ball
        self.field_config_cent = field_config_cent

        self.keypoints = keypoints + kp_ball
        self.kp_ball = kp_ball
        # if self.kp_ball is not None:
        #     self.ball = True 
        # self.skeleton = skeleton
        # self.skeleton_m1 = np.asarray(skeleton) - 1
        self.out_skeleton = out_skeleton
        self.confidence_scales = confidence_scales
        self.nms = nms

        self.timers = defaultdict(float)

        self.adaptive_max_pool_th = adaptive_max_pool_th
        self.max_pool_th = float(max_pool_th)
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~` max pool th ~~~~~~~~~~~~~~~', self.max_pool_th)
        self.decode_masks_first = decode_masks_first
        self.only_output_17 = only_output_17
        self.disable_pred_filter = disable_pred_filter
        self.dec_filter_smaller_than = dec_filter_smaller_than
        self.dec_filter_less_than = dec_filter_less_than

        # init by_target and by_source
        # self.by_target = defaultdict(dict)
        # for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
        #     self.by_target[j2][j1] = (caf_i, True)
        #     self.by_target[j1][j2] = (caf_i, False)
        # self.by_source = defaultdict(dict)
        # for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
        #     self.by_source[j1][j2] = (caf_i, True)
        #     self.by_source[j2][j1] = (caf_i, False)

    def __call__(self, fields, initial_annotations=None, debug=None):
        if self.field_config_cent is not None:
            cif, pan, cif_ball, cif_cent = fields
        else:
            cif, pan, cif_ball = fields
            
        semantic, offsets = pan['semantic'], pan['offset']

        # Ci, Bi = (17, object()) if self.ball else (17, 18)
        Ci, Bi = (17, 0)

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

        # print(self.field_config)
        # print('pif fields',fields[0].shape)
        cifhr = CifHr(self.field_config).fill(fields)
        cifhr_ball = CifHr(self.field_config_ball).fill(fields)
        if self.field_config_cent is not None:
            cifhr_cent = CifHr(self.field_config_cent).fill(fields)

        # seeds = CifSeeds(cifhr.accumulated, self.field_config).fill(fields)

        # caf_scored = CafScored(cifhr.accumulated, self.field_config, self.skeleton).fill(fields)

        Ñ = None

        def cif_local_max(cif, kernel_size=13, pad=6):
            """Use torch for max pooling"""
            cif = torch.tensor(cif)
            cif_m = torch.max_pool2d(cif[None], kernel_size, stride=1, padding=pad)[0] == cif      #### 7 padding=3
            # cif_m &= cif > 0.1# * cif.max()
            cif_m &= cif > self.max_pool_th
            return np.asarray(cif_m)

        # Get coordinates of keypoints of every type
        # list[K,N_k]
        keypoints_yx = []

        if self.decode_masks_first:
            for i_k, cif in enumerate(cifhr.accumulated):
                if i_k == Ci:
                    keypoints_yx.append(np.stack(np.nonzero(cif_local_max(cif)), axis=-1))
                else:
                    keypoints_yx.append(np.stack(np.nonzero(cif), axis=-1))
        else:
            keypoints_yx = [np.stack(np.nonzero(cif_local_max(cif)), axis=-1)
                            for cif in cifhr.accumulated]

        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(15,15))
        # im = plt.imshow(np.log(cifhr.accumulated[Ci]), cmap='jet')
        # plt.colorbar(im)
        # plt.show()

        ball_fyxv = [np.stack(np.nonzero(cif_local_max(cif, kernel_size=51, pad=25)), axis=-1)
                        for cif in cifhr_ball.accumulated]

        if self.field_config_cent is not None:
            center_yx = [np.stack(np.nonzero(cif_local_max(cif)), axis=-1)
                        for cif in cifhr_cent.accumulated]

            keypoints_yx += center_yx
        # plt.figure(figsize=(15,15))
        # # print('max', (cifhr_ball.accumulated[Bi]).max())
        # # print('min', (cifhr_ball.accumulated[Bi]).min())
        # im = plt.imshow(np.log(cifhr_ball.accumulated[Bi]), cmap='jet')
        # plt.colorbar(im)
        # plt.show()
        # print(keypoints_yx)
        # print(ball_fyxv)
        
        keypoints_yx += ball_fyxv


        if debug is not None:
            debug.update(
                cifhr=cifhr,
                fields=fields,
                keypoints_yx=keypoints_yx,
            )
            if self.field_config_cent is not None:
                debug.update(
                    cifhr_cent=cifhr_cent
                )

        
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
        
        # plt.imshow(offsets_to_colorwheel(offsets[None])[0])
        # plt.show()

        difference = (absolute[Ñ,:,:,:] -                   # [ ,2,H,W]
                      keypoints_yx[Ci][:,:,Ñ,Ñ]             # [I,2, , ]
                      )

        distances2 = np.square(difference).sum(axis=1)      # [I,H,W]
        instances_score = distances2.min(axis=0)
        instances = distances2.argmin(axis=0)               # [H,W]
        # instances += 1  # to make sure the ids start from 1 and not 0
        # plt.imshow(instances)
        # plt.show()

        # For each detected keypoints, get its confidence and instance
        if self.field_config_cent is not None:
            centers_fyxv = [
                (Ci, y, x, cifhr_cent.accumulated[0,y,x])
                # (Ci, y, x, 2.)
                for y, x in keypoints_yx[Ci]
                ]
        else:
            centers_fyxv = [
                (Ci, y, x, cifhr.accumulated[Ci,y,x])
                # (Ci, y, x, 2.)
                for y, x in keypoints_yx[Ci]
            ]
        # if self.ball:
        ball_fyxv = [
            (Bi, y, x, cifhr_ball.accumulated[Bi,y,x])
            for y, x in ball_fyxv[Bi]
        ]
        keypoints_fyxiv = [
            (f, y, x, instances[y,x], cifhr.accumulated[f,y,x])
            # (f, y, x, instances[y,x], 2.)
            for f, kp_yx in enumerate(keypoints_yx[:Ci])
            for y, x in kp_yx
        ]

        annotations = []
        for f, y, x, v in centers_fyxv:
            # print('vvvv',v)
            # v = 2.
            annotation = Annotation(
                self.keypoints, self.out_skeleton,
                category_id={17:1,18:37}[f],  # center => person, ball center => ball
                only_output_17=self.only_output_17
                )
            # if not self.only_output_17:
            annotation.add(f, (x,y,v))
            annotations.append(annotation)
        # print('number of centers', len(annotations))

        if not self.decode_masks_first:
            # Assign keypoints to their instance (least confidence first)
            keypoints_fyxiv.sort(key=lambda x:x[-1])
            for f,y,x,i,v in keypoints_fyxiv:
                annotation = annotations[i]
                annotation.add(f, (x,y,v))
                # print('keypoint added')

        # semantic      shape [C,H,W]
        # plt.figure(figsize=(15,15))
        # plt.imshow(semantic[1])
        # plt.show()
        
        # from matplotlib import pyplot as plt
        # print('semantic shape', semantic.shape)
        classes = semantic.argmax(axis=0)   # [H,W]

        
        ###########
        # pad_left, pad_top, pad_right, pad_bottom = 0, 4, 1, 5 # padding copied from logs
        # _, height, width = semantic.shape
        # v_start = pad_top
        # v_stop = height-pad_bottom
        # h_start = pad_left
        # h_stop = width-pad_right


        # ball_from_mask = [np.unravel_index(semantic[2].argmax(), semantic[2].shape)]
        # ball_fyxv_from_mask = [
        #     (Bi, y, x, semantic[2,y,x])
        #     for y, x in ball_from_mask
        # ]


        # plt.imshow(softmax(semantic[None])[0,1])
        # plt.colorbar()
        # plt.show()

        # plt.hist(semantic.reshape(-1))
        # plt.show()
        # plt.savefig('data-mscoco/test.png')

        panoptic = classes*1000 + instances
        # plt.figure(figsize=(20,20))
        # print('show')
        # plt.figure(figsize=(20,20))
        inssss = np.zeros_like(panoptic)
        ids = np.random.permutation(len(annotations))

        if self.decode_masks_first:
            for i in range(len(annotations)):
                annotation = annotations[i]
                centroid_mask = (classes == 1) & (instances == i)
                for f, cif in enumerate(cifhr.accumulated):
                    if f == Ci:
                        continue
                    cif_masked = cif * centroid_mask
                    x, y =np.unravel_index(np.argmax(cif_masked, axis=None), cif_masked.shape)  # argmax retunrs a flat value, we need unravel ind 
                    if cif_masked[x,y] > 0.1:
                        annotation.add(f, (y, x, cif_masked[x,y]))
                    annotation.cls = 1# semantic[:,centroid_mask].sum(axis=1).argmax(axis=0)
                    annotation.mask = centroid_mask
        else:
            for i in range(len(annotations)):
                annotation = annotations[i]
                centroid_mask = (classes == 1) & (instances == i) #& (instances_score > 0.1)
                # print(semantic.shape)
                annotation.cls = 1# semantic[:,centroid_mask].sum(axis=1).argmax(axis=0)
                annotation.mask = centroid_mask
                inssss += (ids[i]+1)*centroid_mask       # to plot instances
        # plt.imshow(inssss)
        # plt.colorbar()
        # plt.show()
        # print('show')

        # if self.ball:
        # if not self.only_output_17:
        for f, y, x, v in ball_fyxv:
            f = 18
            # print('fff', f)
            
            annotation = Annotation(
                        keypoints=self.keypoints,
                        skeleton=self.out_skeleton,
                        category_id=37,
                        only_output_17=self.only_output_17,
                            ).add(f, (x, y, v))
            for ff in range(f):
                annotation.add(ff, (0,0,0))
                annotation.cls = 37# semantic[:,centroid_mask].sum(axis=1).argmax(axis=0)
                annotation.mask = None
            annotations.append(annotation)
        # print('number of centers after ball aded', len(annotations))

        ball_mask = classes == 2
        if ball_mask.sum() > 10:
            pass

        # self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        # if self.force_complete:
        #     annotations = self.complete_annotations(cifhr, fields, annotations)

        # if self.nms is not None:
        #     annotations = self.nms.annotations(annotations)

        if not self.disable_pred_filter:
            # print('not disabled')
            # print('self.dec_filter_smaller_than', self.dec_filter_smaller_than)
            # print('self.dec_filter_less_than', self.dec_filter_less_than)
            filtered_annotations = []
            for ann in annotations:
                if ann.category_id != 1:
                    filtered_annotations.append(ann)
                    continue
                if np.count_nonzero(ann.mask) > self.dec_filter_smaller_than and np.count_nonzero(ann.data[:,2]) >= self.dec_filter_less_than:
                    filtered_annotations.append(ann)
            annotations = filtered_annotations
        
        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        if debug is not None:
            debug.update(
                classes=classes,
                cifhr=cifhr,
                has_ball=self.ball,
                instances=instances,
                instances_score=instances_score,
                keypoints_yx=keypoints_yx,
                keypoints_fyxiv=keypoints_fyxiv,
                centers_fyxv=centers_fyxv,
                annotations=annotations,
                fields=fields,
                panoptic=panoptic
            )
        return annotations
    