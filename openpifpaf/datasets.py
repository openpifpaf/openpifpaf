import copy
import logging
import os
import random

import numpy as np
import torch.utils.data
from PIL import Image

from . import transforms, utils


ANNOTATIONS_TRAIN = 'data-mscoco/annotations/person_keypoints_train2017.json'
ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
DET_ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
DET_ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


class Coco(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
    """

    def __init__(self, root, annFile, *, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=[1],
                 image_filter='keypoint-annotations'):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.category_ids = category_ids

        if image_filter == 'all':
            self.ids = self.coco.getImgIds()
        elif image_filter == 'annotated':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
        elif image_filter == 'keypoint-annotations':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.filter_for_keypoint_annotations()
        else:
            raise Exception('unknown value for image_filter: {}'.format(image_filter))

        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        LOG.info('... done.')

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # mask valid TODO still necessary?
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        LOG.debug(meta)

        # log stats
        for ann in anns:
            if random.random() > 0.1:
                continue
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
        return len(self.ids)


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image = self.images[index].copy().convert('RGB')

        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': 'pilimage{}'.format(index),
        })

        return image, anns, meta

    def __len__(self):
        return len(self.images)


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=None)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=None)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--detection-annotations', default=False, action='store_true')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--loader-workers', default=None, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')

    group_aug = parser.add_argument_group('augmentations')
    group_aug.add_argument('--square-edge', default=385, type=int,
                           help='square edge of input images')
    group_aug.add_argument('--extended-scale', default=False, action='store_true',
                           help='augment with an extended scale range')
    group_aug.add_argument('--orientation-invariant', default=0.0, type=float,
                           help='augment with random orientations')
    group_aug.add_argument('--no-augmentation', dest='augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')


def train_configure(args):
    if args.train_annotations is None:
        args.train_annotations = ANNOTATIONS_TRAIN
        if args.detection_annotations:
            args.train_annotations = DET_ANNOTATIONS_TRAIN
    if args.val_annotations is None:
        args.val_annotations = ANNOTATIONS_VAL
        if args.detection_annotations:
            args.val_annotations = DET_ANNOTATIONS_VAL


def train_preprocess_factory(args):
    if not args.augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.square_edge),
            transforms.CenterPad(args.square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    preprocess_transformations = [
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(), 0.5),
    ]

    assert not (args.extended_scale and args.detection_annotations)
    if args.extended_scale:
        preprocess_transformations.append(
            transforms.RescaleRelative(scale_range=(0.25 * args.rescale_images,
                                                    2.0 * args.rescale_images),
                                       power_law=True)
        )
    elif args.detection_annotations:
        preprocess_transformations.append(
            transforms.RescaleRelative(scale_range=(0.5 * args.rescale_images,
                                                    1.0 * args.rescale_images),
                                       power_law=True)
        )
    else:
        preprocess_transformations.append(
            transforms.RescaleRelative(scale_range=(0.4 * args.rescale_images,
                                                    2.0 * args.rescale_images),
                                       power_law=True)
        )

    preprocess_transformations += [
        # transforms.RandomApply(transforms.ScaleMix(args.square_edge / 2.0), 0.5),
        transforms.Crop(args.square_edge),
        transforms.CenterPad(args.square_edge),
    ]

    if args.orientation_invariant:
        preprocess_transformations += [
            transforms.RandomApply(transforms.RotateBy90(), args.orientation_invariant),
        ]

    preprocess_transformations += [
        transforms.TRAIN_TRANSFORM,
    ]

    return transforms.Compose(preprocess_transformations)


def train_factory(args, target_transforms):
    preprocess = train_preprocess_factory(args)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Coco(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='annotated' if args.detection_annotations else 'keypoint-annotations',
        category_ids=[] if args.detection_annotations else [1],
    )
    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Coco(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='annotated' if args.detection_annotations else 'keypoint-annotations',
        category_ids=[] if args.detection_annotations else [1],
    )
    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader
