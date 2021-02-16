from collections import defaultdict
import copy
import logging
import os
import random

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils
from mlworkflow import PickledDataset, TransformedDataset
from dataset_utilities.ds.instants_dataset import ViewCropperTransform, ExtractViewData
import scipy.ndimage

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class AddBallPositionFactory():
    def __call__(self, view_key, view):
        balls = [a for a in view.annotations if a.type == "ball"]
        ball = balls[0]
        if view_key.camera != ball.camera:
            return {}
        ball_2D = view.calib.project_3D_to_2D(ball.center)
        return {"x": ball_2D.x, "y": ball_2D.y, "visibility": ball.visible}

def build_DeepSportBall_datasets(pickled_dataset_filename, validation_set_size_pc, square_edge):
    dataset = PickledDataset(pickled_dataset_filename)
    keys = list(dataset.keys.all())
    random.shuffle(keys)
    lim = len(keys)*validation_set_size_pc//100
    training_keys = keys[lim:]
    validation_keys = keys[:lim]

    transforms = [
        ViewCropperTransform(output_shape=(square_edge,square_edge), def_min=30, def_max=80, with_diff=False),
        ExtractViewData(
            AddBallPositionFactory()
        )
    ]
    dataset = TransformedDataset(dataset, transforms)
    return DeepSportBalls(dataset, training_keys), DeepSportBalls(dataset, validation_keys)

class DeepSportBalls(torch.utils.data.Dataset):
    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.keys = keys

    def __len__(self):
        return len(self.keys)



    def __getitem__(self, index):
        def build_empty_person(image_id, n_keypoints=17, category_id=1):
            return {
                'num_keypoints': 0,
                'area': 21985.8665, # dummy value
                'iscrowd': 0,
                'keypoints': [0]*3*n_keypoints,
                'image_id': image_id,
                'bbox': [231.34, 160.47, 152.42, 319.53], # dummy values
                'category_id': category_id,
                'id': image_id
            }
        def add_ball_keypoint(ann: dict, x, y, visibility):
            ann["keypoints"] += [x, y, visibility]
            return ann
        key = self.keys[index]
        print(key)
        data = self.dataset.query_item(key)
        image_id = key[0].timestamp
        image = data["input_image"]
        anns = []
        if "x" in data:
            anns = [add_ball_keypoint(build_empty_person(image_id), data["x"], data["y"], data["visibility"])]
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': str(key),
        }

        return image, anns, meta
