import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
import json

from PIL import Image
from .. import transforms, utils

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class VisualRelationship(torch.utils.data.Dataset):

    categories = ['person',
    'sky',
    'building',
    'truck',
    'bus',
    'table',
    'shirt',
    'chair',
    'car',
    'train',
    'glasses',
    'tree',
    'boat',
    'hat',
    'trees',
    'grass',
    'pants',
    'road',
    'motorcycle',
    'jacket',
    'monitor',
    'wheel',
    'umbrella',
    'plate',
    'bike',
    'clock',
    'bag',
    'shoe',
    'laptop',
    'desk',
    'cabinet',
    'counter',
    'bench',
    'shoes',
    'tower',
    'bottle',
    'helmet',
    'stove',
    'lamp',
    'coat',
    'bed',
    'dog',
    'mountain',
    'horse',
    'plane',
    'roof',
    'skateboard',
    'traffic light',
    'bush',
    'phone',
    'airplane',
    'sofa',
    'cup',
    'sink',
    'shelf',
    'box',
    'van',
    'hand',
    'shorts',
    'post',
    'jeans',
    'cat',
    'sunglasses',
    'bowl',
    'computer',
    'pillow',
    'pizza',
    'basket',
    'elephant',
    'kite',
    'sand',
    'keyboard',
    'plant',
    'can',
    'vase',
    'refrigerator',
    'cart',
    'skis',
    'pot',
    'surfboard',
    'paper',
    'mouse',
    'trash can',
    'cone',
    'camera',
    'ball',
    'bear',
    'giraffe',
    'tie',
    'luggage',
    'faucet',
    'hydrant',
    'snowboard',
    'oven',
    'engine',
    'watch',
    'face',
    'street',
    'ramp',
    'suitcase']

    rel_categories = ['on',
    'wear',
    'has',
    'next to',
    'sleep next to',
    'sit next to',
    'stand next to',
    'park next',
    'walk next to',
    'above',
    'behind',
    'stand behind',
    'sit behind',
    'park behind',
    'in the front of',
    'under',
    'stand under',
    'sit under',
    'near',
    'walk to',
    'walk',
    'walk past',
    'in',
    'below',
    'beside',
    'walk beside',
    'over',
    'hold',
    'by',
    'beneath',
    'with',
    'on the top of',
    'on the left of',
    'on the right of',
    'sit on',
    'ride',
    'carry',
    'look',
    'stand on',
    'use',
    'at',
    'attach to',
    'cover',
    'touch',
    'watch',
    'against',
    'inside',
    'adjacent to',
    'across',
    'contain',
    'drive',
    'drive on',
    'taller than',
    'eat',
    'park on',
    'lying on',
    'pull',
    'talk',
    'lean on',
    'fly',
    'face',
    'play with',
    'sleep on',
    'outside of',
    'rest on',
    'follow',
    'hit',
    'feed',
    'kick',
    'skate on']
    def __init__(self, image_dir, ann_file, *, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):

        self.root = image_dir
        self.imgs = [(os.path.join(self.root, k),v) for k,v in json.load(open(ann_file)).items()]


        if n_images:
            self.imgs = self.imgs[:n_images]

        print('Images: {}'.format(len(self.imgs)))

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        chosen_img = self.imgs[index]
        img_path = chosen_img[0]
        with open(os.path.join(img_path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        initial_size = image.size
        meta_init = {
            'dataset_index': index,
            'image_id': index,
            'file_dir': img_path,
            'file_name': os.path.basename(img_path),
        }

        anns = []
        dict_counter = {}
        for target in chosen_img[1]:
            for type_obj in ['subject', 'object']:
                predicate = target['predicate']
                x = target[type_obj]['bbox'][2]
                y = target[type_obj]['bbox'][0]
                w = target[type_obj]['bbox'][3] - x
                h = target[type_obj]['bbox'][1] - y

                x1 = target['object']['bbox'][2]
                y1 = target['object']['bbox'][0]
                w1 = target['object']['bbox'][3] - x1
                h1 = target['object']['bbox'][1] - y1
                if (x, y, w, h) in dict_counter:
                    if type_obj=='subject':
                        if (x1, y1, w1, h1) in dict_counter:
                            anns[dict_counter[(x, y, w, h)]['detection_id']]['object_index'].append(dict_counter[(x1, y1, w1, h1)]['detection_id'])
                        else:
                            anns[dict_counter[(x, y, w, h)]['detection_id']]['object_index'].append(len(anns))
                        anns[dict_counter[(x, y, w, h)]['detection_id']]['predicate'].append(predicate)
                else:
                    object_index = [len(anns) + 1] if (type_obj=='subject') else []
                    if type_obj=='subject':
                        if (x1, y1, w1, h1) in dict_counter:
                            object_index = [dict_counter[(x1, y1, w1, h1)]['detection_id']]
                    dict_counter[(x, y, w, h)] = {'detection_id': len(anns)}
                    anns.append({
                        'detection_id': len(anns),
                        'image_id': index,
                        'category_id': int(target[type_obj]['category']) + 1,
                        'bbox': [x, y, w, h],
                        "area": w*h,
                        "iscrowd": 0,
                        "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                        "segmentation":[],
                        'num_keypoints': 5,
                        'object_index': object_index,
                        'predicate': [predicate] if type_obj=='subject' else [],
                    })
        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.imgs)

    def write_evaluations(self, eval_class, path, total_time):
        pass
