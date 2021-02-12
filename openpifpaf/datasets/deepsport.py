from collections import defaultdict
import copy
import logging
import os
import random

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils
from mlworkflow import PickledDataset

import scipy.ndimage

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


def build_DeepSportBall_datasets(pickled_dataset_filename, validation_set_size_pc):
    dataset = PickledDataset(pickled_dataset_filename)
    keys = list(dataset.keys.all())
    random.shuffle(keys)
    lim = len(keys)*validation_set_size_pc//100
    training_keys = keys[lim:]
    validation_keys = keys[:lim]
    return DeepSportBalls(dataset, training_keys), DeepSportBalls(dataset, validation_keys)

class DeepSportBalls(torch.utils.data.Dataset):
    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.keys = keys

    def __len__(self):
        return len(self.keys)

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

    def __getitem__(self, index):
        key = self.keys[index]
        view = self.dataset.query_item(key)
        # TODO: dataaugmentation
        # TODO: get ball coordinates
        
        image = view.image
        anns = [add_ball_keypoint(build_empty_person(key.timestamp), x, y, visibility)]
        meta = {
            'dataset_index': index,
            'image_id': key.timestamp,
            'file_name': str(key),
        }

        return image, anns, meta
