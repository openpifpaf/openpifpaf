from collections import defaultdict
import copy
import logging
import os
import sys
import random
import imageio

import cv2
import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils
from mlworkflow import PickledDataset, TransformedDataset
from dataset_utilities.ds.instants_dataset import ViewCropperTransform, ExtractViewData #, SetKeypointsOfInterest
import scipy.ndimage
# from dataset_utilities.calib import Point3D

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

BALL_DIAMETER = 23

class AddBallSegmentationTargetViewFactory():
    def __call__(self, view_key, view):
        calib = view.calib
        target = np.zeros((calib.height, calib.width), dtype=np.uint8)
        for ball in [a for a in view.annotations if a.type == "ball" and calib.projects_in(a.center) and a.visible]:
            diameter = calib.compute_length2D(BALL_DIAMETER, ball.center)
            center = calib.project_3D_to_2D(ball.center)
            cv2.circle(target, center.to_int_tuple(), radius=int(diameter/2), color=1, thickness=-1)
        return {
            "mask": target
        }

class AddBallPositionFactory():
    def __call__(self, view_key, view):
        balls = [a for a in view.annotations if a.type == "ball"]
        ball = balls[0]
        if view_key.camera != ball.camera:
            return {}
        ball_2D = view.calib.project_3D_to_2D(ball.center)
        return {"x": ball_2D.x, "y": ball_2D.y, "visible": ball.visible}


class AddHumansSegmentationTargetViewFactory():
    def __call__(self, view_key, view):
        return {"human_masks": view.human_masks}


def build_DeepSportBall_datasets(pickled_dataset_filename, validation_set_size_pc, square_edge, target_transforms, preprocess):
    dataset = PickledDataset(pickled_dataset_filename)
    keys = list(dataset.keys.all())
    random.shuffle(keys)
    lim = len(keys)*validation_set_size_pc//100
    training_keys = keys[lim:]
    validation_keys = keys[:lim]

    # transforms = [
    #     ViewCropperTransform(output_shape=(square_edge,square_edge), def_min=100, def_max=150, on_ball=False, with_diff=False, with_masks=True),
    #     ExtractViewData(
    #         AddBallPositionFactory(),
    #         AddBallSegmentationTargetViewFactory(),
    #         AddHumansSegmentationTargetViewFactory()
    #     )
    # ]
    # dataset = TransformedDataset(dataset, transforms)

    # transforms = [
    #     ViewCropperTransform(
    #     output_shape=(400,400),
    #     def_min=60, def_max=160,
    #     with_masks=True,
    #     keypoint_sampler=SetKeypointsOfInterest(on_player=True)),
    #     ExtractViewData(
    #         AddBallPositionFactory(),
    #         AddBallSegmentationTargetViewFactory(),
    #         AddHumansSegmentationTargetViewFactory()
    #     )
    # ]
    transforms = [
        ViewCropperTransform(output_shape=(400,400), def_min=60, def_max=160, max_angle=8, focus_object="player"),
        ExtractViewData(
            AddBallPositionFactory(),
            AddBallSegmentationTargetViewFactory(),
            AddHumansSegmentationTargetViewFactory()
        )
    ]

    dataset = TransformedDataset(dataset, transforms)

    return \
        DeepSportBalls(dataset, training_keys, target_transforms, preprocess), \
        DeepSportBalls(dataset, validation_keys, target_transforms, preprocess)

class DeepSportBalls(torch.utils.data.Dataset):

    map_categories = {1:1,3:37}

    def __init__(self, dataset, keys, target_transforms=None, preprocess=None):
        self.dataset = dataset
        self.keys = keys
        self.target_transforms = target_transforms
        self.preprocess = preprocess

        self.ball = False
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        def build_empty_person(image_id, n_keypoints=17, category_id=1):
            return {
                'num_keypoints': 0,
                'area': 21985.8665, # dummy value
                'iscrowd': 0,
                'kp_ball': [],
                'keypoints': 3*18*[0],
                'image_id': image_id,
                'bbox': [231.34, 160.47, 152.42, 319.53], # dummy values
                'category_id': category_id,
                'id': image_id
            }
        def add_ball_keypoint(ann: dict, image_shape, x, y, visible, mask):
            height, width, _ = image_shape
            visibility = 2 if visible else 0
            if x < 0 or y < 0 or x >= width or y >= height:
                visiblity = 0
            ann["kp_ball"] += [int(x), int(y), visibility]
            ann["bmask"] = mask
            return ann
        key = self.keys[index]

        data = self.dataset.query_item(key)
        
        if data is None:
            print("Warning: failed to query {}. Using another key.".format(key), file=sys.stderr)
            return self[random.randint(0, len(self)-1)]
        image_id = key[0].timestamp
        image = data["input_image"]
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image)
        # print(np.unique(data['human_masks'], return_counts=True))
        # plt.savefig('test.jpg')
        # print('human masks',data["human_masks"].shape)
        anns = []
        if "x" in data:
            anns = [add_ball_keypoint(build_empty_person(image_id), image.shape, data["x"], data["y"], data["visible"], data["mask"])]
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': str(key),
        }

        annotation = data['human_masks']
        H, W = annotation.shape
        meshgrid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        meshgrid = np.stack(meshgrid, axis=-1)
        
        ## keemotion.py (maxime)
        ins_id, id_c = np.unique(annotation, return_counts=True)
        for instance_id, id_count in zip(ins_id, id_c):
            if instance_id < 1000 or id_count < 1000:
                continue
            # print('count', id_count)
            label = instance_id // 1000
            category_id = self.map_categories[label]

            iid = instance_id % 1000
            mask = annotation == instance_id
            is_crowd = iid == 0

            coords = meshgrid[mask,:]
            center = tuple(coords.mean(axis=0))[::-1]
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            w, h = x2-x1, y2-y1
            x, y = x1+w/2, y1+h/2
            bbox = (x, y, w, h)

            keypoints = np.zeros((18,3))
            if label == 1:
                keypoints[17,:] = (*center, 2)
            # elif label == 3 and self.ball:
            #     keypoints[18,:] = (*center, 2)
            # else:
            #     pass
            # kp_ball = []
            # if self.ball:
            kp_ball = np.zeros((1,3))
            # kp_ball = [0, 0, 0]
            # if label == 3:
            #     kp_ball = [data["x"], data["y"], data["visible"]]

                # raise NotImplementedError('Class label %d'%label)
            # plt.figure()
            # plt.imshow(mask.astype(np.int64))

            anns.append({
                'num_keypoints': 1,
                'area': coords.shape[0],
                'iscrowd': is_crowd,
                'bmask': mask.astype(np.int64),
                'kp_ball': kp_ball,
                'keypoints': keypoints,
                'image_id': str(key),
                'id': instance_id,
                'category_id': category_id,
                'bbox_original': bbox,
                'bbox': bbox,
            })
        

        image = Image.fromarray(image)
        # print(type(anns))
        image, anns, meta = self.preprocess(image, anns, meta)
        # print(type(anns))

        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]
            # anns_dict = dict()
            # for t in self.target_transforms:
            #     # print(type(anns))
            #     ann = t(image, anns, meta)
            #     anns_dict[ann['name']] = ann['value']


        return image, anns, meta


