"""
Interface for custom data.

This module handles datasets and is the class that you need to inherit from for your custom dataset.
This class gives you all the handles so that you can train with a new –dataset=mydataset.
The particular configuration of keypoints and skeleton is specified in the headmeta instances
"""


import argparse
import torch

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from openpifpaf.datasets import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.plugins.coco import CocoDataset as CocoLoader

from .constants import ANIMAL_KEYPOINTS, ANIMAL_SKELETON, HFLIP, \
    ANIMAL_SIGMAS, ANIMAL_POSE, ANIMAL_CATEGORIES, ANIMAL_SCORE_WEIGHTS


class AnimalKp(DataModule):
    """
    Adapted from the standard CocoKp class to work as external plugin
    """

    train_annotations = 'data-animalpose/annotations/animal_keypoints_20_train.json'
    val_annotations = 'data-animalpose/annotations/animal_keypoints_20_val.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-animalpose/images/train/'
    val_image_dir = 'data-animalpose/images/val/'
    eval_image_dir = val_image_dir

    n_images = None
    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()

        cif = headmeta.Cif('cif', 'animal',
                           keypoints=ANIMAL_KEYPOINTS,
                           sigmas=ANIMAL_SIGMAS,
                           pose=ANIMAL_POSE,
                           draw_skeleton=ANIMAL_SKELETON,
                           score_weights=ANIMAL_SCORE_WEIGHTS)
        caf = headmeta.Caf('caf', 'animal',
                           keypoints=ANIMAL_KEYPOINTS,
                           sigmas=ANIMAL_SIGMAS,
                           pose=ANIMAL_POSE,
                           skeleton=ANIMAL_SKELETON)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Animal')

        group.add_argument('--animal-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--animal-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--animal-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--animal-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--animal-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--animal-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--animal-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--animal-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--animal-no-augmentation',
                           dest='animal_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--animal-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--animal-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--animal-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--animal-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')

        # evaluation  (TO setup directly)
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--animal-eval-test2017', default=False, action='store_true')
        eval_set_group.add_argument('--animal-eval-testdev2017', default=False, action='store_true')

        group.add_argument('--animal-no-eval-annotation-filter',
                           dest='animal_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--animal-eval-long-edge', default=cls.eval_long_edge, type=int,
                           dest='animal_eval_long_edge', help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--animal-eval-extended-scale', default=False, action='store_true',
                           dest='animal_eval_extended_scale',)
        group.add_argument('--animal-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float,
                           dest='animal_eval_orientation_invariant')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # Animal specific
        cls.train_annotations = args.animal_train_annotations
        cls.val_annotations = args.animal_val_annotations
        cls.train_image_dir = args.animal_train_image_dir
        cls.val_image_dir = args.animal_val_image_dir

        cls.square_edge = args.animal_square_edge
        cls.extended_scale = args.animal_extended_scale
        cls.orientation_invariant = args.animal_orientation_invariant
        cls.blur = args.animal_blur
        cls.augmentation = args.animal_augmentation  # loaded by the dest name
        cls.rescale_images = args.animal_rescale_images
        cls.upsample_stride = args.animal_upsample
        cls.min_kp_anns = args.animal_min_kp_anns
        cls.b_min = args.animal_bmin

        # evaluation
        cls.eval_annotation_filter = args.animal_eval_annotation_filter
        cls.eval_long_edge = args.animal_eval_long_edge
        cls.eval_orientation_invariant = args.animal_eval_orientation_invariant
        cls.eval_extended_scale = args.animal_eval_extended_scale

        if (args.animal_eval_test2017 or args.animal_eval_testdev2017) \
                and not args.write_predictions \
                and not args.debug:
            raise Exception('have to use --write-predictions for this dataset')

    def _preprocess(self):
        encoders = (encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    encoder.Caf(self.head_metas[1], bmin=self.b_min))

        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                             2.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.3 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RandomApply(
                transforms.HFlip(ANIMAL_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            transforms.RandomApply(
                transforms.Blur(), self.blur),
            transforms.RandomChoice(
                [transforms.RotateBy90(),
                 transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.4],
            ),
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoLoader(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoLoader(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                transforms.DeterministicEqualChoice([
                    transforms.RescaleAbsolute(cls.eval_long_edge),
                    transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return transforms.Compose([
            *self.common_eval_preprocess(),
            transforms.ToAnnotations([
                transforms.ToKpAnnotations(
                    ANIMAL_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                transforms.ToCrowdAnnotations(ANIMAL_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoLoader(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

    def metrics(self):
        return [metric.Coco(
            COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
            keypoint_oks_sigmas=ANIMAL_SIGMAS,
        )]
