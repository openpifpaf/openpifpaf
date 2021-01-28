import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from .. import transforms, utils

import logging
# Swap function
def swapPositions(list, pos1, pos2):

    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class NightOwls(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Based on `torchvision.dataset.CocoDetection`.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    train_image_dir = "data/nightowls/nightowls_validation/"
    val_image_dir = "data/nightowls/nightowls_validation/"
    train_annotations = "data/nightowls/nightowls_validation.json"
    val_annotations = "data/nightowls/nightowls_validation.json"
    test_path = {'val': "data/nightowls/annotations/nightowls_test.json"}

    categories = ['pedestrian', 'bicycledriver', 'motorbikedriver']
    def __init__(self, image_dir, ann_file, *, target_transforms=None, class_ids=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        from pycocotools.coco import COCO
        self.root = image_dir
        self.coco = COCO(ann_file)

        # Image ID
        if class_ids:
            self.ids = []
            for id in class_ids:
                self.ids.extend(list(self.coco.getImgIds(catIds=[id])))
            # Remove duplicates
            self.ids = list(set(self.ids))
        else:
            # All images
            self.ids = list(self.coco.imgs.keys())
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

        # Cat ID (missing class)
        if class_ids:
            self.cat_ids = class_ids
            self.cat_ids = swapPositions(self.cat_ids, 0, self.cat_ids.index(19))
        else:
            self.cat_ids = self.coco.getCatIds()
        print("Number of classes: {}".format(len(self.cat_ids)))
        self.catID_label = {catid:label for label, catid in enumerate(self.cat_ids)}

        # index = 4
        # import pdb; pdb.set_trace()
        # image_id = self.ids[index]
        # ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        # anns = self.coco.loadAnns(ann_ids)
        #
        # anns = copy.deepcopy(anns)
        # for ann in anns:
        #     x, y, w, h = ann["bbox"]
        #     if ann["category_id"] < 4:
        #         ann["iscrowd"] = 0
        #         ann["category_id"] = ann["category_id"]-1
        #         ann["keypoints"] = [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]
        #         ann["segmentation"] = []
        #         ann['num_keypoints'] = 5
        #     else:
        #         ann["iscrowd"] = 1
        #         ann["category_id"] = 0
        #         for catg in range(len(self.categories[1:])):
        #             ann_temp = copy.deepcopy(ann)
        #             ann_temp["category_id"] = catg + 2
        #             ann["keypoints"] = [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]
        #             ann['num_keypoints'] = 5
        #             anns.append(ann_temp)
        #
        # image_info = self.coco.loadImgs(image_id)[0]
        #
        # LOG.debug(image_info)
        # with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
        #     image = Image.open(f).convert('RGB')
        #
        # meta_init = {
        #     'dataset_index': index,
        #     'image_id': image_id,
        #     'file_dir': os.path.join(self.root, image_info['file_name']),
        #     'file_name': image_info['file_name'],
        # }
        #
        # if 'flickr_url' in image_info:
        #     _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
        #     flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
        #     meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)
        #
        # # preprocess image and annotations
        # image, anns, meta = self.preprocess(image, anns, None)
        # meta.update(meta_init)
        #
        # # mask valid
        # valid_area = meta['valid_area']
        # utils.mask_valid_area(image, valid_area)
        #
        # # if there are not target transforms, done here
        # LOG.debug(meta)
        # # transform targets
        # if self.target_transforms is not None:
        #     anns = [t(image, anns, meta) for t in self.target_transforms]
        #
        # return image, anns, meta

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)

        anns = copy.deepcopy(anns)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if ann["category_id"] < 4:
                ann["iscrowd"] = 0
                ann["keypoints"] = [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]
                ann["segmentation"] = []
                ann['num_keypoints'] = 5
            else:
                ann["iscrowd"] = 1
                ann["category_id"] = 1
                ann["keypoints"] = [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]
                ann["segmentation"] = []
                ann['num_keypoints'] = 5
                for catg in range(len(self.categories[1:])):
                    ann_temp = copy.deepcopy(ann)
                    ann_temp["keypoints"] = [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]
                    ann_temp["category_id"] = catg + 2
                    ann_temp["segmentation"] = []
                    ann_temp['num_keypoints'] = 5
                    anns.append(ann_temp)

        image_info = self.coco.loadImgs(image_id)[0]

        LOG.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_dir': os.path.join(self.root, image_info['file_name']),
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)
        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.ids)
