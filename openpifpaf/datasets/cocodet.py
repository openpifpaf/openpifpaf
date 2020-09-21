import argparse

import torch

from .module import DataModule
from .. import encoder, headmeta, metric, transforms
from .coco import Coco
from .cocokp import CocoKp
from .collate import collate_images_anns_meta, collate_images_targets_meta
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    HFLIP,
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class CocoDet(DataModule):
    # cli configurable
    train_annotations = 'data-mscoco/annotations/instances_train2017.json'
    val_annotations = 'data-mscoco/annotations/instances_val2017.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'
    eval_image_dir = val_image_dir

    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    eval_annotation_filter = True

    def __init__(self):
        super().__init__()
        cifdet = headmeta.CifDet('cifdet', 'cocodet', COCO_CATEGORIES)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoDet')

        group.add_argument('--cocodet-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cocodet-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cocodet-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cocodet-val-image-dir',
                           default=cls.val_image_dir)

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

        group.add_argument('--cocodet-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocodet specific
        cls.train_annotations = args.cocodet_train_annotations
        cls.val_annotations = args.cocodet_val_annotations
        cls.train_image_dir = args.cocodet_train_image_dir
        cls.val_image_dir = args.cocodet_val_image_dir

        cls.square_edge = args.cocodet_square_edge
        cls.extended_scale = args.cocodet_extended_scale
        cls.orientation_invariant = args.cocodet_orientation_invariant
        cls.augmentation = args.cocodet_augmentation
        cls.rescale_images = args.cocodet_rescale_images
        cls.upsample_stride = args.cocodet_upsample

        cls.eval_annotation_filter = args.coco_eval_annotation_filter

    def _preprocess(self):
        enc = encoder.CifDet(self.head_metas[0])

        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders([enc]),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.5 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        orientation_t = None
        if self.orientation_invariant:
            orientation_t = transforms.RandomApply(
                transforms.RotateBy90(), self.orientation_invariant)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.AnnotationJitter(),
            transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            orientation_t,
            transforms.MinSize(min_side=4.0),
            transforms.UnclippedArea(threshold=0.75),
            # transforms.UnclippedSides(),
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders([enc]),
        ])

    def train_loader(self):
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self):
        val_data = Coco(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    @staticmethod
    def _eval_preprocess():
        return transforms.Compose([
            *CocoKp.common_eval_preprocess(),
            transforms.ToAnnotations([
                transforms.ToDetAnnotations(COCO_CATEGORIES),
                transforms.ToCrowdAnnotations(COCO_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = Coco(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

    def metrics(self):
        return [metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=100,
            category_ids=[],
            iou_type='bbox',
        )]
