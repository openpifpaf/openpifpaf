import torch

from ..data_module import DataModule
from ..network import headmeta
from .. import transforms
from .coco import Coco
from .collate import collate_images_targets_meta
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    HFLIP,
)


class CocoDet(DataModule):
    description = 'COCO Detection data module.'

    # cli configurable
    train_annotations = 'data-mscoco/annotations/instances_train2017.json'
    val_annotations = 'data-mscoco/annotations/instances_val2017.json'
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'

    n_images = None
    loader_workers = None
    batch_size = 8

    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    augmentation = True
    rescale_images = 1.0

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('data module CocoDet')

        group.add_argument('--cocodet-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cocodet-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cocodet-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cocodet-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--cocodet-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--cocodet-loader-workers',
                           default=cls.loader_workers, type=int,
                           help='number of workers for data loading')
        group.add_argument('--cocodet-batch-size',
                           default=cls.batch_size, type=int,
                           help='batch size')

        group.add_argument('--cocodet-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--cocodet-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--cocodet-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        assert cls.augmentation
        group.add_argument('--cocodet-no-augmentation',
                           dest='cocodet_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--cocodet-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

    @classmethod
    def configure(cls, args):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocokp specific
        cls.train_annotations = args.cocodet_train_annotations
        cls.val_annotations = args.cocodet_val_annotations
        cls.train_image_dir = args.cocodet_train_image_dir
        cls.val_image_dir = args.cocodet_val_image_dir

        cls.n_images = args.cocodet_n_images
        cls.loader_workers = args.cocodet_loader_workers
        cls.batch_size = args.cocodet_batch_size

        cls.square_edge = args.cocodet_square_edge
        cls.extended_scale = args.cocodet_extended_scale
        cls.orientation_invariant = args.cocodet_orientation_invariant
        cls.augmentation = args.cocodet_augmentation
        cls.rescale_images = args.cocodet_rescale_images

        if cls.loader_workers is None:
            cls.loader_workers = cls.batch_size

    def head_metas(self):
        return (headmeta.Detection('cifdet', COCO_CATEGORIES),)

    @classmethod
    def _preprocess(cls):
        if not cls.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(cls.square_edge),
                transforms.CenterPad(cls.square_edge),
                transforms.EVAL_TRANSFORM,
            ])

        if cls.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.5 * cls.rescale_images,
                             2.0 * cls.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.7 * cls.rescale_images,
                             1.5 * cls.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        orientation_t = None
        if cls.orientation_invariant:
            orientation_t = transforms.RandomApply(
                transforms.RotateBy90(), cls.orientation_invariant)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.AnnotationJitter(),
            transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            transforms.Crop(cls.square_edge, use_area_of_interest=True),
            transforms.CenterPad(cls.square_edge),
            orientation_t,
            transforms.MinSize(min_side=4.0),
            transforms.UnclippedArea(),
            transforms.UnclippedSides(),
            transforms.TRAIN_TRANSFORM,
        ])

    def train_loader(self, target_transforms):
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            target_transforms=target_transforms,
            n_images=self.n_images,
            image_filter='annotated',
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self, target_transforms):
        val_data = Coco(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            target_transforms=target_transforms,
            n_images=self.n_images,
            image_filter='annotated',
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)
