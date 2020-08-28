import torch

from .module import DataModule
from ..network import headmeta
from .. import encoder, transforms
from .coco import Coco
from .collate import collate_images_targets_meta
from .constants import (
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)


class CocoKp(DataModule):
    description = 'COCO Keypoint data module.'

    # cli configurable
    train_annotations = 'data-mscoco/annotations/person_keypoints_train2017.json'
    val_annotations = 'data-mscoco/annotations/person_keypoints_val2017.json'
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'

    n_images = None
    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    augmentation = True
    rescale_images = 1.0

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('data module CocoKp')

        group.add_argument('--cocokp-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cocokp-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cocokp-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cocokp-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--cocokp-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--cocokp-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--cocokp-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--cocokp-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        assert cls.augmentation
        group.add_argument('--cocokp-no-augmentation',
                           dest='cocokp_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--cocokp-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

    @classmethod
    def configure(cls, args):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocokp specific
        cls.train_annotations = args.cocokp_train_annotations
        cls.val_annotations = args.cocokp_val_annotations
        cls.train_image_dir = args.cocokp_train_image_dir
        cls.val_image_dir = args.cocokp_val_image_dir

        cls.n_images = args.cocokp_n_images
        cls.square_edge = args.cocokp_square_edge
        cls.extended_scale = args.cocokp_extended_scale
        cls.orientation_invariant = args.cocokp_orientation_invariant
        cls.augmentation = args.cocokp_augmentation
        cls.rescale_images = args.cocokp_rescale_images

    @staticmethod
    def head_metas():
        cif = headmeta.Intensity('cif',
                                 COCO_KEYPOINTS,
                                 COCO_PERSON_SIGMAS,
                                 COCO_UPRIGHT_POSE,
                                 COCO_PERSON_SKELETON)
        caf = headmeta.Association('caf',
                                   COCO_KEYPOINTS,
                                   COCO_PERSON_SIGMAS,
                                   COCO_UPRIGHT_POSE,
                                   COCO_PERSON_SKELETON)
        dcaf = headmeta.Association('caf25',
                                    COCO_KEYPOINTS,
                                    COCO_PERSON_SIGMAS,
                                    COCO_UPRIGHT_POSE,
                                    DENSER_COCO_PERSON_CONNECTIONS,
                                    sparse_skeleton=COCO_PERSON_SKELETON,
                                    only_in_field_of_view=True)
        return cif, caf, dcaf

    @classmethod
    def _preprocess(cls, base_stride):
        metas = cls.head_metas()
        encoders = (
            encoder.Cif(metas[0], base_stride // metas[0].upsample_stride),
            encoder.Caf(metas[1], base_stride // metas[1].upsample_stride),
            encoder.Caf(metas[2], base_stride // metas[2].upsample_stride),
        )

        if not cls.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(cls.square_edge),
                transforms.CenterPad(cls.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if cls.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.25 * cls.rescale_images,
                             2.0 * cls.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.4 * cls.rescale_images,
                             2.0 * cls.rescale_images),
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
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])

    def train_loader(self, base_stride):
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(base_stride),
            n_images=self.n_images,
            image_filter='keypoint-annotations',
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self, base_stride):
        val_data = Coco(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(base_stride),
            n_images=self.n_images,
            image_filter='keypoint-annotations',
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)
