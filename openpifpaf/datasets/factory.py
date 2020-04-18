import torch

from .coco import Coco
from .collate import collate_images_targets_meta
from .. import transforms

ANNOTATIONS_TRAIN = 'data-mscoco/annotations/person_keypoints_train2017.json'
ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
DET_ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
DET_ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=None)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=None)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--dataset', default='cocokp', choices=('cocokp', 'cocodet'))
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
        if args.dataset == 'cocokp':
            args.train_annotations = ANNOTATIONS_TRAIN
        elif args.dataset == 'cocodet':
            args.train_annotations = DET_ANNOTATIONS_TRAIN
        else:
            raise NotImplementedError

    if args.val_annotations is None:
        if args.dataset == 'cocokp':
            args.val_annotations = ANNOTATIONS_VAL
        elif args.dataset == 'cocodet':
            args.val_annotations = DET_ANNOTATIONS_VAL
        else:
            raise NotImplementedError


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

    assert not (args.extended_scale and args.dataset == 'cocodet')
    if args.extended_scale:
        preprocess_transformations.append(
            transforms.RescaleRelative(scale_range=(0.25 * args.rescale_images,
                                                    2.0 * args.rescale_images),
                                       power_law=True)
        )
    elif args.dataset == 'cocodet':
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
        image_filter='keypoint-annotations' if args.dataset == 'cocokp' else 'annotated',
        category_ids=[1] if args.dataset == 'cocokp' else [],
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
        image_filter='keypoint-annotations' if args.dataset == 'cocokp' else 'annotated',
        category_ids=[1] if args.dataset == 'cocokp' else [],
    )
    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader
