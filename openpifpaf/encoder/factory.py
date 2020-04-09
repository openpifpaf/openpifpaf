import logging
import re

from .annrescaler import AnnRescaler, AnnRescalerDet
from .caf import Caf
from .cif import Cif
from .cifdet import CifDet
from .. import visualizer

from ..data import (COCO_KEYPOINTS, COCO_PERSON_SKELETON, COCO_PERSON_SIGMAS, COCO_UPRIGHT_POSE,
                    DENSER_COCO_PERSON_SKELETON,
                    KINEMATIC_TREE_SKELETON, DENSER_COCO_PERSON_CONNECTIONS,
                    COCO_CATEGORIES)

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('CIF encoder')
    group.add_argument('--cif-side-length', default=Cif.side_length, type=int,
                       help='side length of the CIF field')

    group = parser.add_argument_group('CAF encoder')
    group.add_argument('--caf-min-size', default=Caf.min_size, type=int,
                       help='min side length of the CAF field')
    group.add_argument('--caf-fixed-size', default=Caf.fixed_size, action='store_true',
                       help='fixed caf size')
    group.add_argument('--caf-aspect-ratio', default=Caf.aspect_ratio, type=float,
                       help='CAF width relative to its length')


def configure(args):
    # configure CIF
    Cif.side_length = args.cif_side_length

    # configure CAF
    Caf.min_size = args.caf_min_size
    Caf.fixed_size = args.caf_fixed_size
    Caf.aspect_ratio = args.caf_aspect_ratio


def factory(headnames, strides):
    if isinstance(headnames[0], (list, tuple)):
        return [factory(task_headnames, task_strides)
                for task_headnames, task_strides in zip(headnames, strides)]

    encoders = [factory_head(head_name, stride)
                for head_name, stride in zip(headnames, strides)]
    return encoders


def factory_head(head_name, stride):
    cif_m = re.match('[cp]if([0-9]*)$', head_name)
    if cif_m is not None:
        n_keypoints = int(cif_m.group(1)) if cif_m.group(1) else 17
        LOG.debug('using %d keypoints for CIF', n_keypoints)

        LOG.info('selected encoder CIF for %s with %d keypoints', head_name, n_keypoints)
        vis = visualizer.Cif(head_name,
                             stride=stride,
                             keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
        return Cif(AnnRescaler(stride, n_keypoints, COCO_UPRIGHT_POSE),
                   sigmas=COCO_PERSON_SIGMAS,
                   visualizer=vis)

    cifdet_m = re.match('[cp]ifdet([0-9]*)$', head_name)
    if cifdet_m is not None:
        n_categories = int(cifdet_m.group(1)) if cifdet_m.group(1) else 91
        LOG.debug('using %d categories for CIFDET', n_categories)

        LOG.info('selected encoder CIFDET for %s with %d categories', head_name, n_categories)
        vis = visualizer.CifDet(head_name,
                                stride=stride,
                                categories=COCO_CATEGORIES)
        return CifDet(n_categories,
                      AnnRescalerDet(stride, n_categories),
                      visualizer=vis)

    if head_name in ('paf', 'paf19', 'caf', 'wpaf', 'pafb',
                     'paf16',
                     'paf44',
                     'paf25', 'caf25'):
        n_keypoints = 17
        sparse_skeleton = None
        only_in_field_of_view = False
        if head_name in ('paf', 'paf19', 'caf', 'wpaf', 'pafb'):
            skeleton = COCO_PERSON_SKELETON
        elif head_name in ('paf16',):
            skeleton = KINEMATIC_TREE_SKELETON
        elif head_name in ('paf44',):
            skeleton = DENSER_COCO_PERSON_SKELETON
        elif head_name in ('paf25', 'caf25'):
            skeleton = DENSER_COCO_PERSON_CONNECTIONS
            sparse_skeleton = COCO_PERSON_SKELETON
            only_in_field_of_view = True
        else:
            raise Exception('unknown skeleton type of head')

        LOG.info('selected encoder CAF for %s', head_name)
        vis = visualizer.Caf(head_name,
                             stride=stride,
                             keypoints=COCO_KEYPOINTS, skeleton=skeleton)
        return Caf(AnnRescaler(stride, n_keypoints, COCO_UPRIGHT_POSE),
                   skeleton=skeleton,
                   sigmas=COCO_PERSON_SIGMAS,
                   sparse_skeleton=sparse_skeleton,
                   only_in_field_of_view=only_in_field_of_view,
                   visualizer=vis)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
