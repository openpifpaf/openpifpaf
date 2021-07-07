import argparse

import torch
import numpy as np

import openpifpaf

from openpifpaf.plugins.coco import CocoDataset as Coco
from .wholebody_metric import WholebodyMetric
from .constants import (
    COCO_CATEGORIES,
    WHOLEBODY_KEYPOINTS,
    WHOLEBODY_SKELETON,
    WHOLEBODY_SIGMAS,
    WHOLEBODY_SCORE_WEIGHTS,
    WHOLEBODY_STANDING_POSE,
    HFLIP,
    training_weights_local_centrality
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class Wholebody(openpifpaf.datasets.DataModule):
    # Coco wholebody has no test dataset

    # cli configurable
    train_annotations = ('data-mscoco/annotations/'
                         'person_keypoints_train2017_wholebody_pifpaf_style.json')
    val_annotations = ('data-mscoco/annotations/'
                       'coco_wholebody_val_v1.0.json')
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017'

    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 1.0

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False
    weights = None

    def __init__(self):
        super().__init__()
        if self.weights is not None:
            caf_weights = []
            for bone in WHOLEBODY_SKELETON:
                caf_weights.append(max(self.weights[bone[0] - 1],
                                       self.weights[bone[1] - 1]))
            w_np = np.array(caf_weights)
            caf_weights = list(w_np / np.sum(w_np) * len(caf_weights))
        else:
            caf_weights = None
        cif = openpifpaf.headmeta.Cif('cif', 'wholebody',
                                      keypoints=WHOLEBODY_KEYPOINTS,
                                      sigmas=WHOLEBODY_SIGMAS,
                                      pose=WHOLEBODY_STANDING_POSE,
                                      draw_skeleton=WHOLEBODY_SKELETON,
                                      score_weights=WHOLEBODY_SCORE_WEIGHTS,
                                      training_weights=self.weights)
        caf = openpifpaf.headmeta.Caf('caf', 'wholebody',
                                      keypoints=WHOLEBODY_KEYPOINTS,
                                      sigmas=WHOLEBODY_SIGMAS,
                                      pose=WHOLEBODY_STANDING_POSE,
                                      skeleton=WHOLEBODY_SKELETON,
                                      training_weights=caf_weights)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module wholebody')

        group.add_argument('--wholebody-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--wholebody-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--wholebody-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--wholebody-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--wholebody-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--wholebody-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--wholebody-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--wholebody-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--wholebody-no-augmentation',
                           dest='wholebody_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--wholebody-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--wholebody-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--wholebody-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--wholebody-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')
        group.add_argument('--wholebody-apply-local-centrality-weights',
                           dest='wholebody_apply_local_centrality',
                           default=False, action='store_true',
                           help='Weigh the CIF and CAF head during training.')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--wholebody-eval-test2017', default=False,
                                    action='store_true')
        eval_set_group.add_argument('--wholebody-eval-testdev2017', default=False,
                                    action='store_true')

        assert cls.eval_annotation_filter
        group.add_argument('--wholebody-no-eval-annotation-filter',
                           dest='wholebody_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--wholebody-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--wholebody-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--wholebody-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # wholebody specific
        cls.train_annotations = args.wholebody_train_annotations
        cls.val_annotations = args.wholebody_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.wholebody_train_image_dir
        cls.val_image_dir = args.wholebody_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.wholebody_square_edge
        cls.extended_scale = args.wholebody_extended_scale
        cls.orientation_invariant = args.wholebody_orientation_invariant
        cls.blur = args.wholebody_blur
        cls.augmentation = args.wholebody_augmentation
        cls.rescale_images = args.wholebody_rescale_images
        cls.upsample_stride = args.wholebody_upsample
        cls.min_kp_anns = args.wholebody_min_kp_anns
        cls.bmin = args.wholebody_bmin

        # evaluation
        cls.eval_annotation_filter = args.wholebody_eval_annotation_filter
        cls.eval_long_edge = args.wholebody_eval_long_edge
        cls.eval_orientation_invariant = args.wholebody_eval_orientation_invariant
        cls.eval_extended_scale = args.wholebody_eval_extended_scale

        if args.wholebody_apply_local_centrality:
            cls.weights = training_weights_local_centrality
        else:
            cls.weights = None

    def _preprocess(self):
        encoders = (openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(WHOLEBODY_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.RotateBy90(), self.orientation_invariant),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = Coco(
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
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    COCO_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(COCO_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = Coco(
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
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [
            WholebodyMetric(
                self.eval_annotations,
                max_per_image=20,
                category_ids=[1],
                iou_type='keypoints',
            ),
        ]
