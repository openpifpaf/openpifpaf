import argparse

import torch

import openpifpaf

from . import datasets, metric
from .posetrack2018 import Posetrack2018

from .constants import (
    KEYPOINTS,
    SIGMAS,
    UPRIGHT_POSE,
    SKELETON,
    DENSER_CONNECTIONS,
)


class Posetrack2017(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = 'data-posetrack2017/annotations/train/*.json'
    val_annotations = 'data-posetrack2017/annotations/val/*.json'
    eval_annotations = val_annotations
    data_root = 'data-posetrack2017'

    def __init__(self):
        super().__init__()

        cif = openpifpaf.headmeta.TSingleImageCif(
            'cif', 'posetrack2018',
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            draw_skeleton=SKELETON,
        )
        caf = openpifpaf.headmeta.TSingleImageCaf(
            'caf', 'posetrack2018',
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            skeleton=SKELETON,
        )
        dcaf = openpifpaf.headmeta.TSingleImageCaf(
            'dcaf', 'posetrack2018',
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            skeleton=DENSER_CONNECTIONS,
            sparse_skeleton=SKELETON,
            only_in_field_of_view=True,
        )
        tcaf = openpifpaf.headmeta.Tcaf(
            'tcaf', 'posetrack2018',
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            draw_skeleton=SKELETON,
            only_in_field_of_view=True,
        )

        cif.upsample_stride = Posetrack2018.upsample_stride
        caf.upsample_stride = Posetrack2018.upsample_stride
        dcaf.upsample_stride = Posetrack2018.upsample_stride
        tcaf.upsample_stride = Posetrack2018.upsample_stride
        self.head_metas = [cif, caf, dcaf, tcaf]

        if Posetrack2018.ablation_without_tcaf:
            self.head_metas = [cif, caf, dcaf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Posetrack2017')

        group.add_argument('--posetrack2017-eval-annotations',
                           default=cls.eval_annotations,
                           help='eval annotations')
        group.add_argument('--posetrack2017-data-root',
                           default=cls.data_root,
                           help='data root')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # posetrack2017 specific
        cls.eval_annotations = args.posetrack2017_eval_annotations
        cls.data_root = args.posetrack2017_data_root

    def train_loader(self):
        raise NotImplementedError

    def val_loader(self):
        raise NotImplementedError

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *Posetrack2018.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    ['person'],
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(['person']),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = datasets.Posetrack2017(
            annotation_files=self.eval_annotations,
            data_root=self.data_root,
            preprocess=self._eval_preprocess(),
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        return openpifpaf.datasets.LoaderWithReset(eval_loader, 'annotation_file')

    def metrics(self):
        eval_data = datasets.Posetrack2017(
            annotation_files=self.eval_annotations,
            data_root=self.data_root,
            preprocess=self._eval_preprocess(),
        )
        return [metric.Posetrack(
            images=eval_data.meta_images(),
            categories=eval_data.meta_categories(),
            ground_truth=self.eval_annotations,
            output_format='2017',
        )]
