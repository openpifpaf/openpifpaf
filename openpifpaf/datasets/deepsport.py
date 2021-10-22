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


def niels_split(keys):
    l = [(f'{k.instant_key.arena_label }_{k.instant_key.game_id}_{k.instant_key.timestamp}',k) for k in keys]
    l = sorted(l, key=lambda kv: kv[0])
    return {
        "testing": [x[1] for x in l[-100:]],
        "validation": [x[1] for x in l[-200:-100]],
        "training": [x[1] for x in l[:-200]]
    }


class DeepSportKeysSplitter(): # pylint: disable=too-few-public-methods
    def __init__(self, validation_pc=15, eval_frequency=5):
        self.validation_pc = validation_pc
        self.eval_frequency = eval_frequency
    @staticmethod
    def split_equally(d, K):
        """
            splits equally the keys of d given their values
            arguments:
                d: a dict {"label1": 30, "label2": 45, "label3": 22, ... "label20": 14}
            returns:
                a list of list splitting equally their value:
                [[label1, label12, label19], [label2, label15], [label3, label10, label11], ...]
        """
        s = sorted(d.items(), key=lambda kv: kv[1])
        f = [{"count": 0, "list": []} for _ in range(K)]
        while s:
            arena_label, count = s.pop(-1)
            index, _ = min(enumerate(f), key=(lambda x: x[1]["count"]))
            f[index]["count"] += count
            f[index]["list"].append(arena_label)
        return [x["list"] for x in f]

    @staticmethod
    def count_keys_per_arena_label(keys):
        """returns a dict of (arena_label: number of keys of that arena)"""
        bins = {}
        for key in keys:
            bins[key.arena_label] = bins.get(key.arena_label, 0) + 1
        return bins
    @staticmethod
    def count_keys_per_game_id(keys):
        """returns a dict of (game_id: number of keys of that game)"""
        bins = {}
        for key in keys:
            bins[key.game_id] = bins.get(key.game_id, 0) + 1
        return bins
    def __call__(self, keys, fold=0):
        split = {
            "A": ['KS-FR-CAEN', 'KS-FR-LIMOGES', 'KS-FR-ROANNE'],
            "B": ['KS-FR-NANTES', 'KS-FR-BLOIS', 'KS-FR-FOS'],
            "C": ['KS-FR-LEMANS', 'KS-FR-MONACO', 'KS-FR-STRASBOURG'],
            "D": ['KS-FR-GRAVELINES', 'KS-FR-STCHAMOND', 'KS-FR-POITIERS'],
            "E": ['KS-FR-NANCY', 'KS-FR-BOURGEB', 'KS-FR-VICHY'],
            "F": ['KS-US-RUTGERS', 'KS-CH-FRIBOURG'],
            "G": ['KS-AT-VIENNA', 'KS-FI-KOTKA'],
            "H": ['KS-FI-ESPOO', 'KS-BE-MONS'],
            "I": ['KS-BE-OSTENDE', 'KS-FI-TAMPERE', 'KS-FI-LAPUA'],
            "J": ['KS-FI-SALO', 'KS-AT-KLOSTERNEUBURG', 'KS-BE-SPIROU'],
            "K": ['KS-FI-FORSSA', 'KS-US-IPSWICH'],
        }
        assert 0 <= fold <= len(split)-1, "Invalid fold index"
        fold = chr(ord("A")+fold)
        testing_keys = [k for k in keys if k.arena_label in split[fold]]
        remaining_keys = [k for k in keys if k not in testing_keys]

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        validation_keys = random.sample(remaining_keys, len(keys)*self.validation_pc//100)

        # Restore random seed
        random.setstate(random_state)

        training_keys = [k for k in remaining_keys if k not in validation_keys]

        return {
            "training": training_keys,
            "validation": validation_keys,
            "testing": testing_keys,
        }

class KFoldsTestingKeysSplitter(DeepSportKeysSplitter):
    def __init__(self, fold_count=8, validation_pc=15):
        self.fold_count = fold_count
        self.validation_pc = validation_pc

    def __call__(self, keys, fold=0):
        assert fold >= 0 and fold < self.fold_count

        keys_dict = self.count_keys_per_arena_label(keys)
        keys_lists = self.split_equally(keys_dict, self.fold_count)

        testing_keys = [k for k in keys if k.arena_label in keys_lists[fold]]
        remaining_keys = [k for k in keys if k not in testing_keys]

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        validation_keys = random.sample(remaining_keys, len(keys)*self.validation_pc//100)

        # Restore random seed
        random.setstate(random_state)

        training_keys = [k for k in remaining_keys if k not in validation_keys]

        return {
            "training": training_keys,
            "validation": validation_keys,
            "testing": testing_keys
        }

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
        size = view.calib.compute_length2D(BALL_DIAMETER, ball.center)
        return {"x": ball_2D.x, "y": ball_2D.y, "visible": ball.visible, "size": size}

class AddHumansSegmentationTargetViewFactory():
    def __call__(self, view_key, view):
        if not hasattr(view, "human_masks"):
            view.human_masks = np.zeros(view.image.shape[0:2])
        return {"human_masks": view.human_masks}




def deepsportlab_dataset_splitter(keys, method=None, fold=0, validation_set_size_pc=None):
    print(f"splitting the dataset with '{method}' strategy")
    if method == "Niels":
        split = niels_split(keys)
        training_keys = split["training"]
        testing_keys = split["testing"]
        validation_keys = split["validation"]
        assert 460 < len(training_keys) <= 472, "This split is supposed to occur on the dataset of 661 views"
    elif method == "KFoldTesting":
        sets = KFoldsTestingKeysSplitter(validation_pc=validation_set_size_pc)(keys, fold=fold)
        training_keys = sets["training"]
        validation_keys = sets["validation"]
        testing_keys = sets["testing"]
    elif method == "NoTestSet":
        random_state = random.getstate()
        random.seed(0)
        random.shuffle(keys)
        lim = len(keys)*validation_set_size_pc//100
        training_keys = keys[lim:]
        validation_keys = keys[:lim]
        random.seed(random_state)
        testing_keys = []
    elif method == "DeepSport":
        sets = DeepSportKeysSplitter()(keys, fold=fold)
        training_keys = sets["training"]
        validation_keys = sets["validation"]
        testing_keys = sets["testing"]
    else:
        raise BaseException("method not found")
    return {
        "training": training_keys,
        "validation": validation_keys,
        "testing": testing_keys
    }

def build_DeepSportBall_datasets(pickled_dataset_filename, validation_set_size_pc, square_edge, target_transforms, preprocess, focus_object=None, config=None, dataset_fold=None, debug_on_test=False):
    dataset = PickledDataset(pickled_dataset_filename)

    keys = list(dataset.keys.all())
    if dataset_fold is None:
        method = "NoTestSet"
        fold = 0
    elif dataset_fold == "Niels":
        method = "Niels"
        fold = 0
    elif dataset_fold == "DeepSport":
        method = "DeepSport"
        fold = 0
    else:
        method = "KFoldTesting"
        fold = dataset_fold
    split = deepsportlab_dataset_splitter(keys, method, fold, validation_set_size_pc)

    transforms = [
        ViewCropperTransform(output_shape=(square_edge,square_edge), def_min=30, def_max=80, max_angle=8, focus_object=focus_object),

        ExtractViewData(
            AddBallPositionFactory(),
            AddBallSegmentationTargetViewFactory(),
            AddHumansSegmentationTargetViewFactory(),
        )
    ]

    dataset = TransformedDataset(dataset, transforms)
    if debug_on_test:
        return \
        DeepSportDataset(dataset, split["testing"], target_transforms, preprocess, config), \
        DeepSportDataset(dataset, split["validation"], target_transforms, preprocess, config)    
    return \
        DeepSportDataset(dataset, split["training"], target_transforms, preprocess, config), \
        DeepSportDataset(dataset, split["validation"], target_transforms, preprocess, config)

class DeepSportDataset(torch.utils.data.Dataset):

    map_categories = {1:1,3:37}

    def __init__(self, dataset, keys, target_transforms=None, preprocess=None, config=None, oks_computation=False):
        self.dataset = dataset
        self.keys = keys
        self.target_transforms = target_transforms
        self.preprocess = preprocess
        self.ball = False
        
        self.oks_computation = oks_computation

        print('deepsport config', config)
        if 'ball' in config:
            self.ball = True

        self.config = config[0]
        
        
        print('Number of images deepsport:', len(self.keys))
        LOG.info('Number of images deepsport: %d', len(self.keys))

    def __len__(self):
        return len(self.keys)
        # return 24


    def Get_number_of_images_with_ball(self):
        LOG.info('Images with ball ...')
        def has_ball(data):
            if "x" in data:
                image = data["input_image"]
                x = data['x']
                y = data['y']
                visible = data['visible']
                height, width, _ = image.shape
                visibility = 2 if visible else 0
                if x < 0 or y < 0 or x >= width or y >= height:
                    visibility = 0
                if visibility == 2:
                    return True
            return False

        keys = []
        for key in self.keys:

            data = self.dataset.query_item(key)
            if data is None:
                continue
            if has_ball(data):
                keys.append(key)

        self.keys = keys
        print('Number of images with ball', len(self.keys))
        LOG.info('... done.')

    def __getitem__(self, index):
        def build_empty_person(image_id, n_keypoints=17, category_id=37):
            return {
                'num_keypoints': 0,
                'area': 0, # dummy value
                'iscrowd': 0,
                'keypoints': 3*n_keypoints*[0],
                'cent': 3*[0],
                'image_id': image_id,
                'bbox': [0, 0, 0, 0], # dummy values
                'category_id': category_id,
                'id': image_id,
                'put_nan': True
            }
        def add_ball_keypoint(ann: dict, image_shape, x, y, visible, mask):
            height, width, _ = image_shape
            visiblity = 2 if visible else 0
            if x < 0 or y < 0 or x >= width or y >= height:
                visiblity = 0

            key = "kp_ball"   # Custom CifBall decoding
            
            ann[key] = []
            ann[key].append(int(x))      # add center for y
            ann[key].append(int(y))      # add center for x
            ann[key].append(visiblity)
            ann['bmask'] = np.zeros_like(mask)      # to get the mask of ball from human_masks !!!
            return ann
        key = self.keys[index]
        
        data = self.dataset.query_item(key)
        
        
        if data is None:
            # print("Warning: failed to query {}. Using another key.".format(key), file=sys.stderr)
            # TODO it will be problematic for oks comupatation
            return self[random.randint(0, len(self)-1)]
        image_id = key[0].timestamp
        image = data["input_image"]
        
        anns = []
        n_keypoints = 18 if self.config == 'cifcent' else 17
        if self.ball:
            if "x" in data:
                anns = [add_ball_keypoint(build_empty_person(image_id,n_keypoints=n_keypoints), image.shape, data["x"], data["y"], data["visible"], data["mask"])]
        
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': str(key.instant_key.arena_label)+'_'+str(key.instant_key.game_id)+
                        '_'+str(key.instant_key.timestamp)+'_'+str(key.camera)+'_'+str(key.index),
        }
        
        if "ball_size" in data:
            meta['ball_size'] = data["size"]
        

        if self.config in ['cif', 'cifcent', 'pan']:
            
            annotation = data['human_masks']
            
            H, W = annotation.shape
            meshgrid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            meshgrid = np.stack(meshgrid, axis=-1)
            
            ins_id, id_c = np.unique(annotation, return_counts=True)
            for instance_id, id_count in zip(ins_id, id_c):
                if instance_id < 1000 or id_count < 10:
                    continue
                
                label = instance_id // 1000
                
                if label not in [1, 3]:
                    continue
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

                keypoints = np.zeros((n_keypoints,3))
                kp_ball = np.zeros((1,3))
                if label == 1 and n_keypoints > 17:
                    keypoints[17,:] = (*center, 2)
                

                anns.append({
                    'num_keypoints': 1,
                    'area': coords.shape[0],
                    'iscrowd': is_crowd,
                    'bmask': mask.astype(np.int64),
                    'kp_ball': kp_ball,
                    'keypoints': keypoints,
                    'cent': (*center, 2),
                    'image_id': str(key),
                    'id': instance_id,
                    'category_id': category_id,
                    'bbox_original': bbox,
                    'bbox': bbox,
                    'put_nan': True
                })
        
        image = Image.fromarray(image)
        
        image, anns, meta = self.preprocess(image, anns, meta)
        if False:# self.oks_computation:
            if self.target_transforms is not None:
                anns = [t(image, anns, meta, pq_computation=False) for t in self.target_transforms]
        else:
            if self.target_transforms is not None:
                anns = [t(image, anns, meta) for t in self.target_transforms]
        
        if self.oks_computation:
            return image, anns, meta, data, key      # return the view for oks computation
        return image, anns, meta


