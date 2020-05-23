import torch

from .coco import Coco
from .collate import collate_images_targets_meta
from .constants import COCO_KEYPOINTS, HFLIP
from .. import transforms

COCOKP_ANNOTATIONS_TRAIN = 'data-mscoco/annotations/person_keypoints_train2017.json'
COCOKP_ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
COCODET_ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
COCODET_ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
COCO_IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
COCO_IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--cocokp-train-annotations', default=COCOKP_ANNOTATIONS_TRAIN)
    group.add_argument('--cocodet-train-annotations', default=COCODET_ANNOTATIONS_TRAIN)
    group.add_argument('--cocokp-val-annotations', default=COCOKP_ANNOTATIONS_VAL)
    group.add_argument('--cocodet-val-annotations', default=COCODET_ANNOTATIONS_VAL)
    group.add_argument('--coco-train-image-dir', default=COCO_IMAGE_DIR_TRAIN)
    group.add_argument('--coco-val-image-dir', default=COCO_IMAGE_DIR_VAL)
    group.add_argument('--dataset', default='cocokp')
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


def train_configure(_):
    pass


def train_cocokp_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.25 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.4 * rescale_images, 2.0 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=True),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.TRAIN_TRANSFORM,
    ])


def train_cocodet_preprocess_factory(
        *,
        square_edge,
        augmentation=True,
        extended_scale=False,
        orientation_invariant=0.0,
        rescale_images=1.0,
):
    if not augmentation:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(square_edge),
            transforms.CenterPad(square_edge),
            transforms.EVAL_TRANSFORM,
        ])

    if extended_scale:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.5 * rescale_images, 2.0 * rescale_images),
            power_law=True)
    else:
        rescale_t = transforms.RescaleRelative(
            scale_range=(0.7 * rescale_images, 1.5 * rescale_images),
            power_law=True)

    orientation_t = None
    if orientation_invariant:
        orientation_t = transforms.RandomApply(transforms.RotateBy90(), orientation_invariant)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.AnnotationJitter(),
        transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
        rescale_t,
        transforms.Crop(square_edge, use_area_of_interest=False),
        transforms.CenterPad(square_edge),
        orientation_t,
        transforms.MinSize(min_side=4.0),
        transforms.UnclippedArea(),
        transforms.UnclippedSides(),
        transforms.TRAIN_TRANSFORM,
    ])


def train_cocokp_factory(args, target_transforms):
    preprocess = train_cocokp_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Coco(
        image_dir=args.coco_train_image_dir,
        ann_file=args.cocokp_train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='keypoint-annotations',
        category_ids=[1],
    )
    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Coco(
        image_dir=args.coco_val_image_dir,
        ann_file=args.cocokp_val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='keypoint-annotations',
        category_ids=[1],
    )
    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_cocodet_factory(args, target_transforms):
    preprocess = train_cocodet_preprocess_factory(
        square_edge=args.square_edge,
        augmentation=args.augmentation,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
        rescale_images=args.rescale_images)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    train_data = Coco(
        image_dir=args.coco_train_image_dir,
        ann_file=args.cocodet_train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='annotated',
        category_ids=[],
    )
    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        sampler=torch.utils.data.WeightedRandomSampler(
            train_data.class_aware_sample_weights(), len(train_data), replacement=True),
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    val_data = Coco(
        image_dir=args.coco_val_image_dir,
        ann_file=args.cocodet_val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
        image_filter='annotated',
        category_ids=[],
    )
    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        sampler=torch.utils.data.WeightedRandomSampler(
            val_data.class_aware_sample_weights(), len(val_data), replacement=True),
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader


def train_factory(args, target_transforms):
    if args.dataset in ('cocokp',):
        return train_cocokp_factory(args, target_transforms)
    if args.dataset in ('cocodet',):
        return train_cocodet_factory(args, target_transforms)

    raise Exception('unknown dataset: {}'.format(args.dataset))
