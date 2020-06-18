import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from .. import transforms, utils

import pandas as pd
import glob

LOG = logging.getLogger(__name__)

class UAVDT(torch.utils.data.Dataset):
    """`UAVDT <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    train_image_dir = "data/UAV-benchmark-M/train/"
    val_image_dir = "data/UAV-benchmark-M/test/"
    train_annotations = "data/UAV-benchmark-M/GT/"
    val_annotations = "data/UAV-benchmark-M/GT/"
    test_path = {'val': "data/UAV-benchmark-M/test/"}

    #categories = ["car", "truck", "bus", "van", "cyclist", "pedestrian"]
    categories = ["vehicle"]
    def __init__(self, image_dir, ann_file, *, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        self.root = image_dir
        self.annFile = ann_file
        folders = os.listdir(self.root)

        self.imgs = []
        self.targets = []
        self.targets_ignore = []
        ids = set()
        if self.annFile:
            for folder in folders:
                df = pd.read_csv(os.path.join(self.annFile, folder+'_gt_whole.txt'), sep=',', header=None)
                grouped = df.groupby(0)
                ignore_file = utils.is_non_zero_file(os.path.join(self.annFile, folder+'_gt_ignore.txt'))
                if ignore_file:
                    df_ignore = pd.read_csv(os.path.join(self.annFile, folder+'_gt_ignore.txt'), sep=',', header=None)
                    grouped_ignore = df_ignore.groupby(0)


                for name, group in grouped:
                    if ignore_file and (name in grouped_ignore.groups):
                        group_ignore = grouped_ignore.get_group(name)
                        label_ignore = group_ignore[[8, 2, 3, 4, 5, 6, 7]].values
                        self.targets_ignore.append(label_ignore)
                    else:
                        self.targets_ignore.append([])

                    img_name = 'img{0:06d}.jpg'.format(int(name))
                    image_path = os.path.join(self.root, folder, img_name)
                    self.imgs.append(image_path)
                    label_img = group[[8, 2, 3, 4, 5, 6, 7]].values
                    ids.update(label_img[:,0].tolist())
                    self.targets.append(label_img)
            self.targets = np.asarray(self.targets)
            self.targets_ignore = np.asarray(self.targets_ignore)
        else:
            self.imgs = glob.glob(self.root + '/*/*' + '.jpg')
        self.imgs = np.asarray(self.imgs)
        if n_images:
            choices = np.random.choice(len(self.imgs), n_images, replace=False)
            self.imgs = self.imgs[choices]
            if self.annFile:
                self.targets = self.targets[choices]
                self.targets_ignore = self.targets_ignore[choices]

        ids = set(ids)



        print('Images: {}'.format(len(self.imgs)))

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

        # Cat ID (missing class)
        self.cat_ids = list(ids)
        self.catID_label = {catid:label for label, catid in enumerate(self.cat_ids)}
        #self.PIF_category = PIF_Category(num_classes=len(self.cat_ids), catID_label=self.catID_label)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_path = self.imgs[index]
        with open(os.path.join(img_path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        initial_size = image.size
        meta_init = {
            'dataset_index': index,
            'image_id': index,
            'file_dir': os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]),
            'file_name': os.path.basename(img_path),
        }
        anns = []
        if self.annFile:
            for target in self.targets[index]:
                w = target[3]
                h = target[4]
                x = target[1]
                y = target[2]
                anns.append({
                    'image_id': index,
                    'category_id': 0,#int(target[0]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "segmentation":[],
                    'keypoints': [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    'num_keypoints': 5,
                })
            for target in self.targets_ignore[index]:
                w = target[3]
                h = target[4]
                x = target[1]
                y = target[2]
                anns.append({
                    'image_id': index,
                    'category_id': 0, #int(target[0]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 1,
                    "segmentation":[],
                    'keypoints': [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    'num_keypoints': 5,
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
        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.imgs)

    def write_evaluations(self, eval_class, path, total_time):
        dict_folder = defaultdict(list)
        for annotation in eval_class.predictions:
            x, y, w, h = annotation['bbox']
            categ = int(annotation['category_id'])
            fileName = annotation['file_dir']
            fileName = fileName.split("/")
            folder = fileName[-2]
            image_numb = int(fileName[-1][3:9])
            dict_folder[folder].append(",".join(list(map(str,[image_numb, -1, x, y, w, h, s, 1, categ-1]))))

        for folder in dict_folder.keys():
            utils.mkdir_if_missing(path)
            with open(os.path.join(path,folder+".txt"), "w") as file:
                file.write("\n".join(dict_folder[folder]))
