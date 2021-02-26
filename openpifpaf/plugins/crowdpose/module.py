import argparse
import logging

import numpy as np
import torch

import openpifpaf

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

LOG = logging.getLogger(__name__)


KEYPOINTS = [
    'left_shoulder',  # 1
    'right_shoulder',  # 2
    'left_elbow',  # 3
    'right_elbow',  # 4
    'left_wrist',  # 5
    'right_wrist',  # 6
    'left_hip',  # 7
    'right_hip',  # 8
    'left_knee',  # 9
    'right_knee',  # 10
    'left_ankle',  # 11
    'right_ankle',  # 12
    'head',  # 13
    'neck',  # 14
]
SKELETON = [
    [13, 14],  # head-neck
    [14, 1], [14, 2],  # neck to shoulders
    [1, 2],  # shoulders
    [7, 8],  # hips
    [1, 3], [3, 5],  # left arm
    [2, 4], [4, 6],  # right arm
    [1, 7],  # left shoulder-hip
    [2, 8],  # right shoulder-hip
    [7, 9], [9, 11],  # left leg
    [8, 10], [10, 12],  # right leg
]
# sigmas:
# https://github.com/Jeff-sjtu/CrowdPose/blob/master/
# crowdpose-api/PythonAPI/crowdposetools/cocoeval.py#L223
SIGMAS = [
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
    0.079,  # head
    0.079,  # neck
]
UPRIGHT_POSE = np.array([
    [-1.4, 8.0, 2.0],  # 'left_shoulder',
    [1.4, 8.0, 2.0],  # 'right_shoulder',
    [-1.75, 6.0, 2.0],  # 'left_elbow',
    [1.75, 6.2, 2.0],  # 'right_elbow',
    [-1.75, 4.0, 2.0],  # 'left_wrist',
    [1.75, 4.2, 2.0],  # 'right_wrist',
    [-1.26, 4.0, 2.0],  # 'left_hip',
    [1.26, 4.0, 2.0],  # 'right_hip',
    [-1.4, 2.0, 2.0],  # 'left_knee',
    [1.4, 2.1, 2.0],  # 'right_knee',
    [-1.4, 0.0, 2.0],  # 'left_ankle',
    [1.4, 0.1, 2.0],  # 'right_ankle',
    [0.0, 10.3, 2.0],  # head
    [0.0, 9.3, 2.0],  # neck,
])
HFLIP = openpifpaf.plugins.coco.constants.HFLIP
COCO_CATEGORIES = openpifpaf.plugins.coco.constants.COCO_CATEGORIES


class CrowdPose(openpifpaf.datasets.DataModule):
    _test_annotations = 'data-crowdpose/json/crowdpose_test.json'

    # cli configurable
    train_annotations = 'data-crowdpose/json/crowdpose_train.json'
    val_annotations = 'data-crowdpose/json/crowdpose_val.json'
    eval_annotations = val_annotations
    image_dir = 'data-crowdpose/images/'

    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1

    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False
    eval_crowdpose_index = None

    def __init__(self):
        super().__init__()

        cif = openpifpaf.headmeta.Cif('cif', 'crowdpose',
                                      keypoints=KEYPOINTS,
                                      sigmas=SIGMAS,
                                      pose=UPRIGHT_POSE,
                                      draw_skeleton=SKELETON)
        caf = openpifpaf.headmeta.Caf('caf', 'crowdpose',
                                      keypoints=KEYPOINTS,
                                      sigmas=SIGMAS,
                                      pose=UPRIGHT_POSE,
                                      skeleton=SKELETON)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CrowdPose')

        group.add_argument('--crowdpose-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--crowdpose-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--crowdpose-image-dir',
                           default=cls.image_dir)

        group.add_argument('--crowdpose-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--crowdpose-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--crowdpose-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        assert cls.augmentation
        group.add_argument('--crowdpose-no-augmentation',
                           dest='crowdpose_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--crowdpose-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--crowdpose-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--crowdpose-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--crowdpose-eval-test', default=False, action='store_true')

        group.add_argument('--crowdpose-eval-long-edge', default=cls.eval_long_edge, type=int)
        assert not cls.eval_extended_scale
        group.add_argument('--crowdpose-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--crowdpose-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)
        group.add_argument('--crowdpose-index', choices=('easy', 'medium', 'hard'),
                           default=None)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # crowdpose specific
        cls.train_annotations = args.crowdpose_train_annotations
        cls.val_annotations = args.crowdpose_val_annotations
        cls.image_dir = args.crowdpose_image_dir

        cls.square_edge = args.crowdpose_square_edge
        cls.extended_scale = args.crowdpose_extended_scale
        cls.orientation_invariant = args.crowdpose_orientation_invariant
        cls.augmentation = args.crowdpose_augmentation
        cls.rescale_images = args.crowdpose_rescale_images
        cls.upsample_stride = args.crowdpose_upsample
        cls.min_kp_anns = args.crowdpose_min_kp_anns

        # evaluation
        if args.crowdpose_eval_test:
            cls.eval_annotations = cls._test_annotations
        cls.eval_long_edge = args.crowdpose_eval_long_edge
        cls.eval_orientation_invariant = args.crowdpose_eval_orientation_invariant
        cls.eval_extended_scale = args.crowdpose_eval_extended_scale
        cls.eval_crowdpose_index = args.crowdpose_index

    def _preprocess(self):
        encoders = (openpifpaf.encoder.Cif(self.head_metas[0]),
                    openpifpaf.encoder.Caf(self.head_metas[1]))

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

        orientation_t = None
        if self.orientation_invariant:
            orientation_t = openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.RotateBy90(), self.orientation_invariant)

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.AnnotationJitter(),
            openpifpaf.transforms.RandomApply(openpifpaf.transforms.HFlip(KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            orientation_t,
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = openpifpaf.plugins.coco.CocoDataset(
            image_dir=self.image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = openpifpaf.plugins.coco.CocoDataset(
            image_dir=self.image_dir,
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

    @staticmethod
    def _filter_crowdindex(data: openpifpaf.plugins.coco.CocoDataset, min_index, max_index):
        filtered_ids = []
        for id_ in data.ids:
            image_info = data.coco.imgs[id_]
            LOG.debug('image info %s', image_info)
            crowdindex = image_info['crowdIndex']
            if min_index <= crowdindex < max_index:
                filtered_ids.append(id_)

        LOG.info('crowdindex filter from %d to %d images', len(data.ids), len(filtered_ids))
        data.ids = filtered_ids

    def eval_loader(self):
        eval_data = openpifpaf.plugins.coco.CocoDataset(
            image_dir=self.image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=(self.eval_annotations == self.val_annotations),
            min_kp_anns=self.min_kp_anns if self.eval_annotations == self.val_annotations else 0,
            category_ids=[1],
        )
        if self.eval_crowdpose_index == 'easy':
            self._filter_crowdindex(eval_data, 0.0, 0.1)
        elif self.eval_crowdpose_index == 'medium':
            self._filter_crowdindex(eval_data, 0.1, 0.8)
        elif self.eval_crowdpose_index == 'hard':
            self._filter_crowdindex(eval_data, 0.8, 1.0)

        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
            keypoint_oks_sigmas=SIGMAS,
        )]
