import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
from collections import defaultdict

from PIL import Image
from .. import transforms, utils

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class VisDrone(torch.utils.data.Dataset):
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
    train_image_dir = "data/VisDrone2019/VisDrone2019-DET-train/images"
    val_image_dir = "data/VisDrone2019/VisDrone2019-DET-val/images"
    train_annotations = "data/VisDrone2019/VisDrone2019-DET-train/annotations"
    val_annotations = "data/VisDrone2019/VisDrone2019-DET-val/annotations"

    test_path = {'val': "data/VisDrone2019/VisDrone2019-DET-val/images", 'test-dev-images': "data/VisDrone2019/VisDrone2019-DET-test-dev/images",'test-dev-annotations': "data/VisDrone2019/VisDrone2019-DET-test-dev/annotations", 'test-challenge': "data/VisDrone2019/VisDrone2019-DET-test-challenge/images"}

    categories = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    def __init__(self, image_dir, ann_file, *, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        self.root = image_dir
        self.imgs = [os.path.join(self.root, f) for f in os.listdir(self.root)]

        if n_images:
            self.imgs = self.imgs[:n_images]
        self.targets = []
        self.targets_ignored = []
        ids = set()
        if ann_file:
            for row, img_path in enumerate(self.imgs):
                with open(os.path.splitext(img_path)[0].replace("images", "annotations")+".txt", 'r') as txt:
                    list_temp = []
                    list_temp_ignored = []
                    for line in txt.readlines():
                        line_list = [float(elem) for elem in line.strip().split(",") if elem !='']
                        if line_list[5] != 0.0 and line_list[5] != 11.0:
                            list_temp.append(line_list)
                        else:
                            list_temp_ignored.append(line_list)
                    label_img = np.asarray(list_temp)
                    label_img_ignored = np.asarray(list_temp_ignored)
                    #label_img = np.asarray([np.asarray(list(map(float, line.split(",")))) for line in txt.readlines()])
                    if label_img.shape[0]!=0:
                        ids.update(label_img[:,5].tolist())
                        self.targets.append(label_img)
                        self.targets_ignored.append(label_img_ignored)
                    else:
                        self.imgs = np.delete(self.imgs, row, 0)

        self.targets = np.asarray(self.targets)

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
        img_path = self.imgs[index]
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
        if len(self.targets)>0:
            for target in self.targets[index]:
                w = target[2]
                h = target[3]
                x = target[0]
                y = target[1]
                anns.append({
                    'image_id': index,
                    'category_id': int(target[5]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5
                })
        if len(self.targets_ignored)>0:
            for target in self.targets_ignored[index]:
                w = target[2]
                h = target[3]
                x = target[0]
                y = target[1]
                for i in range(len(self.categories)):
                    anns.append({
                        'image_id': index,
                        'category_id': int(i),
                        'bbox': [x, y, w, h],
                        "area": w*h,
                        "iscrowd": 1,
                        "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                        "segmentation":[],
                        'num_keypoints': 5
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
        dict_folder = defaultdict(list)
        for annotation in eval_class.predictions:
            x, y, w, h = annotation['bbox']
            categ = int(annotation['category_id'])
            fileName = annotation['file_dir']
            fileName = fileName.split("/")
            folder = os.path.splitext(fileName[-1])[0]
            dict_folder[folder].append(",".join(list(map(str,[x, y, w, h, annotation['score'], categ, -1, -1]))))

        for folder in dict_folder.keys():
            utils.mkdir_if_missing(path)
            with open(os.path.join(path,folder+".txt"), "w") as file:
                file.write("\n".join(dict_folder[folder]))
