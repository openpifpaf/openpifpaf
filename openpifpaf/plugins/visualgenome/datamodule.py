import argparse
import numpy as np
import torch

import openpifpaf

from .visual_genome import VisualGenome
from .constants import BBOX_KEYPOINTS, BBOX_HFLIP, OBJ_CATEGORIES, REL_CATEGORIES
from . import metric
from .transforms import Crop


class VGModule(openpifpaf.datasets.DataModule):
    data_dir = "data/visual_genome/"

    debug = False
    pin_memory = False

    n_images = -1
    square_edge = [513]
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    special_preprocess = False
    max_long_edge = False
    no_flipping = False
    use_dcn = False
    supervise_offset = False
    only_det = True
    ignore_rel = False
    caf_bigger = False
    refine_feature = False
    laplace = False
    detach_deform = False
    detach_offset = False
    remove_duplicate = True
    filter_non_overlap = False
    hadamard_product = False

    eval_long_edge = None
    eval_orientation_invariant = 0.0
    eval_extended_scale = False
    vg_512 = False

    def __init__(self):
        super().__init__()

        cifdet = openpifpaf.headmeta.CifDet('cifdet', 'visualgenome', OBJ_CATEGORIES)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Visual Genome')

        group.add_argument('--visualgenome-data-dir',
                           default=cls.data_dir)

        group.add_argument('--visualgenome-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--visualgenome-square-edge',
                           default=cls.square_edge, type=int, nargs='+',
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--visualgenome-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--visualgenome-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--visualgenome-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--visualgenome-no-augmentation',
                           dest='visualgenome_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--visualgenome-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--visualgenome-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--visualgenome-special-preprocess',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--visualgenome-max-long-edge',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--visualgenome-no-flipping',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--visualgenome-use-dcn',
                           default=False, action='store_true',
                           help='use deformable Conv in head')
        group.add_argument('--visualgenome-supervise-offset',
                           default=False, action='store_true',
                           help='Supervise offset of deformable Conv in head')
        group.add_argument('--visualgenome-ignore-rel',
                           default=False, action='store_true',
                           help='ignore relationship everywhere')

        group.add_argument('--visualgenome-det',
                           default=cls.only_det, action='store_true',
                           help='only detection')
        group.add_argument('--visualgenome-use-512',
                           default=False, action='store_true',
                           help='only detection')
        group.add_argument('--visualgenome-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet_CN')
        group.add_argument('--visualgenome-refine-feature',
                           default=False, action='store_true',
                           help='Refine full feature')
        group.add_argument('--visualgenome-laplace',
                           default=False, action='store_true',
                           help='Use heads with laplace')
        group.add_argument('--visualgenome-detach-offset',
                           default=False, action='store_true',
                           help='Detach offset training')
        group.add_argument('--visualgenome-detach-deform',
                           default=False, action='store_true',
                           help='Detach deform training')
        group.add_argument('--visualgenome-no-filter-nonoverlap',
                           default=False, action='store_true',
                           help='Filter out objects with no overlap')
        group.add_argument('--visualgenome-keep-duplicate',
                           default=False, action='store_true',
                           help='Keep many relationships between objects')
        group.add_argument('--visualgenome-hadamard-refined',
                           default=False, action='store_true',
                           help='hadamard Product to refine (not concat)')
        group.add_argument('--visualgenome-caf-bigger',
                           default=False, action='store_true',
                           help='Use RAF CAF Bigger')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # visual genome specific
        cls.data_dir = args.visualgenome_data_dir
        cls.n_images = args.visualgenome_n_images
        cls.square_edge = args.visualgenome_square_edge
        if len(cls.square_edge) == 1:
            cls.square_edge = cls.square_edge[0]
        cls.extended_scale = args.visualgenome_extended_scale
        cls.orientation_invariant = args.visualgenome_orientation_invariant
        cls.blur = args.visualgenome_blur
        cls.augmentation = args.visualgenome_augmentation
        cls.rescale_images = args.visualgenome_rescale_images
        cls.upsample_stride = args.visualgenome_upsample
        cls.special_preprocess = args.visualgenome_special_preprocess
        cls.max_long_edge = args.visualgenome_max_long_edge
        cls.no_flipping = args.visualgenome_no_flipping
        cls.caf_bigger = args.visualgenome_caf_bigger

        cls.use_dcn = args.visualgenome_use_dcn
        cls.supervise_offset = args.visualgenome_supervise_offset

        cls.only_det = args.visualgenome_det
        cls.ignore_rel = args.visualgenome_ignore_rel
        cls.vg_512 = args.visualgenome_use_512
        cls.cifdet_cn = args.visualgenome_cifdet
        cls.refine_feature = args.visualgenome_refine_feature
        cls.laplace = args.visualgenome_laplace
        cls.detach_deform = args.visualgenome_detach_deform
        cls.detach_offset = args.visualgenome_detach_offset
        cls.filter_non_overlap = not args.visualgenome_no_filter_nonoverlap
        cls.remove_duplicate = not args.visualgenome_keep_duplicate
        cls.hadamard_product = args.visualgenome_hadamard_refined

    @staticmethod
    def _convert_data(parent_data, meta):
        image, category_id = parent_data

        anns = [{
            'bbox': np.asarray([5, 5, 21, 21], dtype=np.float32),
            'category_id': category_id + 1,
        }]

        return image, anns, meta

    def _preprocess(self):
        # encoders = (openpifpaf.encoder.CifDet(self.head_metas[0]),
        #             Raf(self.head_metas[1]),)
        encoders = (openpifpaf.encoder.CifDet(self.head_metas[0]),)

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.5 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        orientation_t = None
        if self.orientation_invariant:
            orientation_t = openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.RotateBy90(), self.orientation_invariant)

        if self.special_preprocess:
            return openpifpaf.transforms.Compose([
                #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RandomApply(openpifpaf.transforms.HFlip(BBOX_KEYPOINTS, BBOX_HFLIP), 0.5),
                #rescale_t,
                openpifpaf.transforms.RescaleRelative(scale_range=(0.8 * self.rescale_images,
                             1.3* self.rescale_images), absolute_reference=self.square_edge),
                openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
                openpifpaf.transforms.CenterPad(self.square_edge),
                orientation_t,
                #openpifpaf.transforms.MinSize(min_side=4.0),
                openpifpaf.transforms.UnclippedArea(threshold=0.8),
                # transforms.UnclippedSides(),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.no_flipping:
            return openpifpaf.transforms.Compose([
                #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.NormalizeAnnotations(),
                #rescale_t,
                #openpifpaf.transforms.RescaleRelative(scale_range=(0.7 * self.rescale_images,
                #             1.5* self.rescale_images), absolute_reference=self.square_edge),
                openpifpaf.transforms.RescaleRelative(scale_range=(0.7 * self.rescale_images,
                             1.5* self.rescale_images)),
                openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
                openpifpaf.transforms.CenterPad(self.square_edge),
                #orientation_t,
                openpifpaf.transforms.MinSize(min_side=4.0),
                openpifpaf.transforms.UnclippedArea(threshold=0.75),
                # transforms.UnclippedSides(),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.max_long_edge:
            return openpifpaf.transforms.Compose([
                #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RandomApply(openpifpaf.transforms.HFlip(BBOX_KEYPOINTS, BBOX_HFLIP), 0.5),
                #rescale_t,
                openpifpaf.transforms.RescaleRelative(scale_range=(0.7 * self.rescale_images,
                             1* self.rescale_images), absolute_reference=self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                orientation_t,
                #openpifpaf.transforms.MinSize(min_side=4.0),
                # transforms.UnclippedSides(),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        return openpifpaf.transforms.Compose([
            #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
            openpifpaf.transforms.NormalizeAnnotations(),
            #openpifpaf.transforms.AnnotationJitter(),
            openpifpaf.transforms.RandomApply(openpifpaf.transforms.HFlip(BBOX_KEYPOINTS, BBOX_HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            #openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            orientation_t,
            #openpifpaf.transforms.MinSize(min_side=4.0),
            openpifpaf.transforms.UnclippedArea(threshold=0.75),
            # transforms.UnclippedSides(),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = VisualGenome(
            data_dir=self.data_dir,
            preprocess=self._preprocess(),
            num_im=self.n_images,
            use_512=self.vg_512,
            filter_non_overlap=self.filter_non_overlap,
            filter_duplicate_rels=self.remove_duplicate,
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = VisualGenome(
            data_dir=self.data_dir,
            preprocess=self._preprocess(),
            num_im=5000,  # self.n_images,
            split='test',
            use_512=self.vg_512,
        )

        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def _eval_preprocess(self):
        rescale_t = None
        if self.eval_extended_scale:
            assert self.eval_long_edge
            rescale_t = openpifpaf.transforms.DeterministicEqualChoice([
                openpifpaf.transforms.RescaleAbsolute(self.eval_long_edge),
                openpifpaf.transforms.RescaleAbsolute((self.eval_long_edge) // 2),
            ], salt=1)
        elif self.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(self.eval_long_edge)
        padding_t = None
        if self.batch_size == 1:
            #padding_t = openpifpaf.transforms.CenterPadTight(128)
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert self.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(self.eval_long_edge)

        orientation_t = None
        if self.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToDetAnnotations(OBJ_CATEGORIES),
                openpifpaf.transforms.ToCrowdAnnotations(OBJ_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = VisualGenome(
            data_dir=self.data_dir,
            preprocess=self._eval_preprocess(),
            num_im=self.n_images,
            split='test',
            use_512=self.vg_512,
            eval_mode=True,
        )

        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [
            metric.VG(obj_categories=OBJ_CATEGORIES, rel_categories=REL_CATEGORIES,
                      mode='sgdet', iou_types=['bbox'] if (not self.only_det) else ['bbox']),
            metric.VG(obj_categories=OBJ_CATEGORIES, rel_categories=REL_CATEGORIES,
                      mode='sgcls', iou_types=['bbox'] if (not self.only_det) else ['bbox']),
            metric.VG(obj_categories=OBJ_CATEGORIES, rel_categories=REL_CATEGORIES,
                      mode='predcls', iou_types=['bbox'] if (not self.only_det) else ['bbox'])
        ]
