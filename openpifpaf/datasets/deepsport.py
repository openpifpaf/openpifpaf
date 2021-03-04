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
from dataset_utilities.ds.instants_dataset import ViewCropperTransform, ExtractViewData
import scipy.ndimage

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

    transforms = [
        ViewCropperTransform(output_shape=(square_edge,square_edge), def_min=80, def_max=160, on_ball=False, with_diff=False, with_masks=True),
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
    def __init__(self, dataset, keys, target_transforms=None, preprocess=None):
        self.dataset = dataset
        self.keys = keys
        self.target_transforms = target_transforms
        self.preprocess = preprocess
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        def build_empty_person(image_id, n_keypoints=17, category_id=1):
            return {
                'num_keypoints': 0,
                'area': 21985.8665, # dummy value
                'iscrowd': 0,
                'keypoints': [],
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
            ann["keypoints"] += [int(x), int(y), visibility]
            ann["bmask"] = mask
            return ann
        key = self.keys[index]

        data = self.dataset.query_item(key)
        if data is None:
            print("Warning: failed to query {}. Using another key.".format(key), file=sys.stderr)
            return self[random.randint(0, len(self)-1)]
        image_id = key[0].timestamp
        image = data["input_image"]
        # print('human masks',data["human_masks"])
        anns = []
        if "x" in data:
            anns = [add_ball_keypoint(build_empty_person(image_id), image.shape, data["x"], data["y"], data["visible"], data["mask"])]
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': str(key),
        }


        image = Image.fromarray(image)
        image, anns, meta = self.preprocess(image, anns, meta)

        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]



        return image, anns, meta


