import argparse
import numpy as np
import torch
import torchvision

import openpifpaf

from .visual_genome import VG
from .constants import BBOX_KEYPOINTS, BBOX_HFLIP, OBJ_CATEGORIES, REL_CATEGORIES, REL_CATEGORIES_FLIP
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
    obj_categories = OBJ_CATEGORIES
    rel_categories = REL_CATEGORIES
    vg_512 = False

    def __init__(self):
        super().__init__()
        self.head_metas = []
        cifdet = openpifpaf.headmeta.CifDet('cifdet', 'vg', self.obj_categories)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas.append(cifdet)

        print("Chosen HeadMetas: ", [type(hn) for hn in self.head_metas])
    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Visual Genome')

        group.add_argument('--vg-data-dir',
                           default=cls.data_dir)

        group.add_argument('--vg-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--vg-square-edge',
                           default=cls.square_edge, type=int, nargs='+',
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--vg-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--vg-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--vg-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--vg-no-augmentation',
                           dest='vg_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--vg-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--vg-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--vg-special-preprocess',
                           dest='vg_special_preprocess',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--vg-max-long-edge',
                           dest='vg_max_long_edge',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--vg-no-flipping',
                           dest='vg_no_flipping',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--vg-use-dcn',
                dest='vg_use_dcn',
                default=False, action='store_true',
                help='use deformable Conv in head')
        group.add_argument('--vg-supervise-offset',
                dest='vg_supervise_offset',
                default=False, action='store_true',
                help='Supervise offset of deformable Conv in head')
        group.add_argument('--vg-ignore-rel',
                dest='vg_ignore_rel',
                default=False, action='store_true',
                help='ignore relationship everywhere')

        group.add_argument('--vg-det',
                dest='vg_det',
                default=cls.only_det, action='store_true',
                help='only detection')
        group.add_argument('--vg-use-512',
                dest='vg_512',
                default=False, action='store_true',
                help='only detection')
        group.add_argument('--vg-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet_CN')
        group.add_argument('--vg-refine-feature',
                           default=False, action='store_true',
                           help='Refine full feature')
        group.add_argument('--vg-laplace',
                           default=False, action='store_true',
                           help='Use heads with laplace')
        group.add_argument('--vg-detach-offset',
                           default=False, action='store_true',
                           help='Detach offset training')
        group.add_argument('--vg-detach-deform',
                           default=False, action='store_true',
                           help='Detach deform training')
        group.add_argument('--vg-no-filter-nonoverlap',
                           default=False, action='store_true',
                           help='Filter out objects with no overlap')
        group.add_argument('--vg-keep-duplicate',
                           default=False, action='store_true',
                           help='Keep many relationships between objects')
        group.add_argument('--vg-hadamard-refined',
                           default=False, action='store_true',
                           help='hadamard Product to refine (not concat)')
        group.add_argument('--vg-caf-bigger',
                dest='vg_caf_bigger',
                default=False, action='store_true',
                help='Use RAF CAF Bigger')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # visual genome specific
        # cls.train_annotations = args.vg_train_annotations
        # cls.val_annotations = args.vg_val_annotations
        # cls.train_image_dir = args.vg_train_image_dir
        # cls.val_image_dir = args.vg_val_image_dir
        # cls.eval_image_dir = cls.val_image_dir
        # cls.eval_annotations = cls.val_annotations
        cls.data_dir = args.vg_data_dir
        cls.n_images = args.vg_n_images
        cls.square_edge = args.vg_square_edge
        if len(cls.square_edge)==1:
            cls.square_edge = cls.square_edge[0]
        cls.extended_scale = args.vg_extended_scale
        cls.orientation_invariant = args.vg_orientation_invariant
        cls.blur = args.vg_blur
        cls.augmentation = args.vg_augmentation
        cls.rescale_images = args.vg_rescale_images
        cls.upsample_stride = args.vg_upsample
        cls.special_preprocess = args.vg_special_preprocess
        cls.max_long_edge = args.vg_max_long_edge
        cls.no_flipping = args.vg_no_flipping
        cls.caf_bigger = args.vg_caf_bigger

        cls.use_dcn = args.vg_use_dcn
        cls.supervise_offset = args.vg_supervise_offset

        cls.only_det = args.vg_det
        cls.ignore_rel = args.vg_ignore_rel
        cls.vg_512 = args.vg_512
        cls.cifdet_cn = args.vg_cifdet
        cls.refine_feature = args.vg_refine_feature
        cls.laplace = args.vg_laplace
        cls.detach_deform = args.vg_detach_deform
        cls.detach_offset = args.vg_detach_offset
        cls.filter_non_overlap = not args.vg_no_filter_nonoverlap
        cls.remove_duplicate = not args.vg_keep_duplicate
        cls.hadamard_product = args.vg_hadamard_refined

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
        train_data = VG(
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
        val_data = VG(
            data_dir=self.data_dir,
            preprocess=self._preprocess(),
            num_im=5000,#self.n_images,
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
                openpifpaf.transforms.ToDetAnnotations(self.obj_categories),
                openpifpaf.transforms.ToCrowdAnnotations(self.obj_categories),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])
    def _get_fg_matrix(self):
        # train_data = VisualRelationship(
        #     image_dir=self.train_image_dir,
        #     ann_file=self.train_annotations
        # )
        train_data = VG(
            data_dir=self.data_dir,
        )


        self.head_metas[-1].fg_matrix, self.head_metas[-1].bg_matrix, self.head_metas[-1].smoothing_pred = train_data.get_frequency_prior(self.obj_categories, self.rel_categories)


    def eval_loader(self):
        eval_data = VG(
            data_dir=self.data_dir,
            preprocess=self._eval_preprocess(),
            num_im=self.n_images,
            split='test',
            use_512=self.vg_512,
            eval_mode=True,
        )
        if not self.only_det:
            self._get_fg_matrix()

        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        eval_data = VG(
            data_dir=self.data_dir,
            preprocess=self._eval_preprocess(),
            num_im=self.n_images,
            split='test',
            use_512=self.vg_512,
            eval_mode=True,
        )

        # eval_data = None

        return [metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='sgdet', iou_types=['bbox'] if (not self.only_det) else ['bbox'], vg_eval=eval_data),
                metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='sgcls', iou_types=['bbox'] if (not self.only_det) else ['bbox'], vg_eval=eval_data),
                metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='predcls', iou_types=['bbox'] if (not self.only_det) else ['bbox'], vg_eval=eval_data)
                ]
