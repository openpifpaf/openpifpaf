import argparse

import torch

import openpifpaf
from openpifpaf.plugins.coco.constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_PERSON_SCORE_WEIGHTS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)
from openpifpaf.transforms.pair import SingleImage as S

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class CocoKpSt(openpifpaf.datasets.DataModule):
    max_shift = 30.0

    def __init__(self):
        super().__init__()

        cif = openpifpaf.headmeta.TSingleImageCif(
            'cif', 'cocokpst',
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            draw_skeleton=COCO_PERSON_SKELETON,
            score_weights=COCO_PERSON_SCORE_WEIGHTS,
        )
        caf = openpifpaf.headmeta.TSingleImageCaf(
            'caf', 'cocokpst',
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            skeleton=COCO_PERSON_SKELETON,
        )
        dcaf = openpifpaf.headmeta.TSingleImageCaf(
            'caf25', 'cocokpst',
            keypoints=COCO_KEYPOINTS,
            sigmas=COCO_PERSON_SIGMAS,
            pose=COCO_UPRIGHT_POSE,
            skeleton=DENSER_COCO_PERSON_CONNECTIONS,
            sparse_skeleton=COCO_PERSON_SKELETON,
            only_in_field_of_view=True,
        )
        tcaf = openpifpaf.headmeta.Tcaf(
            'tcaf', 'cocokpst',
            keypoints_single_frame=COCO_KEYPOINTS,
            sigmas_single_frame=COCO_PERSON_SIGMAS,
            pose_single_frame=COCO_UPRIGHT_POSE,
            draw_skeleton_single_frame=COCO_PERSON_SKELETON,
            only_in_field_of_view=True,
        )

        cif.upsample_stride = openpifpaf.plugins.coco.CocoKp.upsample_stride
        caf.upsample_stride = openpifpaf.plugins.coco.CocoKp.upsample_stride
        dcaf.upsample_stride = openpifpaf.plugins.coco.CocoKp.upsample_stride
        tcaf.upsample_stride = openpifpaf.plugins.coco.CocoKp.upsample_stride
        self.head_metas = ([cif, caf, dcaf, tcaf]
                           if openpifpaf.plugins.coco.CocoKp.with_dense
                           else [cif, caf, tcaf])

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoKpSt')
        group.add_argument('--cocokpst-max-shift',
                           default=cls.max_shift, type=float,
                           help='max shift')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.max_shift = args.cocokpst_max_shift

    def _preprocess(self):
        bmin = openpifpaf.plugins.coco.CocoKp.bmin
        encoders = (
            openpifpaf.encoder.SingleImage(openpifpaf.encoder.Cif(self.head_metas[0], bmin=bmin)),
            openpifpaf.encoder.SingleImage(openpifpaf.encoder.Caf(self.head_metas[1], bmin=bmin)),
            openpifpaf.encoder.SingleImage(openpifpaf.encoder.Caf(self.head_metas[2], bmin=bmin)),
            openpifpaf.encoder.Tcaf(self.head_metas[3], bmin=bmin),
        ) if len(self.head_metas) == 4 else (
            openpifpaf.encoder.SingleImage(openpifpaf.encoder.Cif(self.head_metas[0], bmin=bmin)),
            openpifpaf.encoder.SingleImage(openpifpaf.encoder.Caf(self.head_metas[1], bmin=bmin)),
            openpifpaf.encoder.Tcaf(self.head_metas[2], bmin=bmin),
        )

        if not openpifpaf.plugins.coco.CocoKp.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(openpifpaf.plugins.coco.CocoKp.square_edge),
                openpifpaf.transforms.CenterPad(openpifpaf.plugins.coco.CocoKp.square_edge),
                openpifpaf.transforms.pair.ImageToTracking(),
                S(openpifpaf.transforms.EVAL_TRANSFORM),
                openpifpaf.transforms.pair.Encoders(encoders),
            ])

        if openpifpaf.plugins.coco.CocoKp.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * openpifpaf.plugins.coco.CocoKp.rescale_images,
                             2.0 * openpifpaf.plugins.coco.CocoKp.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * openpifpaf.plugins.coco.CocoKp.rescale_images,
                             2.0 * openpifpaf.plugins.coco.CocoKp.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.pair.ImageToTracking(),
            openpifpaf.transforms.RandomApply(openpifpaf.transforms.pair.RandomizeOneFrame(), 0.2),
            S(openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5)),
            S(rescale_t),
            S(openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(angle_perturbation=30.0, prepad=True),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [openpifpaf.plugins.coco.CocoKp.orientation_invariant, 0.4],
            )),
            openpifpaf.transforms.pair.Crop(
                openpifpaf.plugins.coco.CocoKp.square_edge, max_shift=self.max_shift),
            openpifpaf.transforms.pair.Pad(
                openpifpaf.plugins.coco.CocoKp.square_edge, max_shift=self.max_shift),
            S(openpifpaf.transforms.RandomChoice([openpifpaf.transforms.Blur(),
                                                  openpifpaf.transforms.HorizontalBlur()],
                                                 [openpifpaf.plugins.coco.CocoKp.blur / 2.0,
                                                  openpifpaf.plugins.coco.CocoKp.blur / 2.0])),
            S(openpifpaf.transforms.TRAIN_TRANSFORM),
            openpifpaf.transforms.pair.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = openpifpaf.plugins.coco.CocoDataset(
            image_dir=openpifpaf.plugins.coco.CocoKp.train_image_dir,
            ann_file=openpifpaf.plugins.coco.CocoKp.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=openpifpaf.plugins.coco.CocoKp.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size // 2,
            shuffle=(not openpifpaf.plugins.coco.CocoKp.debug
                     and openpifpaf.plugins.coco.CocoKp.augmentation),
            pin_memory=openpifpaf.plugins.coco.CocoKp.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_tracking_images_targets_meta,
        )

    def val_loader(self):
        val_data = openpifpaf.plugins.coco.CocoDataset(
            image_dir=openpifpaf.plugins.coco.CocoKp.val_image_dir,
            ann_file=openpifpaf.plugins.coco.CocoKp.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=openpifpaf.plugins.coco.CocoKp.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size // 2,
            shuffle=False,
            pin_memory=openpifpaf.plugins.coco.CocoKp.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_tracking_images_targets_meta,
        )

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *openpifpaf.plugins.coco.CocoKp.common_eval_preprocess(),
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
        eval_data = openpifpaf.plugins.coco.CocoDataset(
            image_dir=openpifpaf.plugins.coco.CocoKp.eval_image_dir,
            ann_file=openpifpaf.plugins.coco.CocoKp.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=openpifpaf.plugins.coco.CocoKp.eval_annotation_filter,
            min_kp_anns=(openpifpaf.plugins.coco.CocoKp.min_kp_anns
                         if openpifpaf.plugins.coco.CocoKp.eval_annotation_filter
                         else 0),
            category_ids=[1] if openpifpaf.plugins.coco.CocoKp.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=openpifpaf.plugins.coco.CocoKp.pin_memory,
            num_workers=self.loader_workers,
            drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta,
        )

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(openpifpaf.plugins.coco.CocoKp.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
        )]
