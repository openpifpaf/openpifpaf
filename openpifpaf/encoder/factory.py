import logging
import re

from .paf import Caf
from .pif import Pif
from .skeleton import Skeleton
from .visualizer import Visualizer

from ..data import (COCO_PERSON_SKELETON, COCO_PERSON_SIGMAS, DENSER_COCO_PERSON_SKELETON,
                    KINEMATIC_TREE_SKELETON, DENSER_COCO_PERSON_CONNECTIONS)

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('CIF encoder')
    group.add_argument('--cif-side-length', default=Pif.side_length, type=int,
                       help='side length of the CIF field')

    group = parser.add_argument_group('CAF encoder')
    group.add_argument('--caf-min-size', default=Caf.min_size, type=int,
                       help='min side length of the CAF field')
    group.add_argument('--caf-fixed-size', default=Caf.fixed_size, action='store_true',
                       help='fixed caf size')
    group.add_argument('--caf-aspect-ratio', default=Caf.aspect_ratio, type=float,
                       help='CAF width relative to its length')

    group = parser.add_argument_group('debug')
    group.add_argument('--debug-cif-indices', default=[], nargs='+', type=int,
                       help='indices of CIF fields to create debug plots for')
    group.add_argument('--debug-caf-indices', default=[], nargs='+', type=int,
                       help='indices of CAF fields to create debug plots for')
    group.add_argument('--debug-dcaf-indices', default=[], nargs='+', type=int,
                       help='indices of dense CAF fields to create debug plots for')


def configure(args):
    # configure Pif
    Pif.side_length = args.cif_side_length

    # configure Caf
    Caf.min_size = args.caf_min_size
    Caf.fixed_size = args.caf_fixed_size
    Caf.aspect_ratio = args.caf_aspect_ratio

    # configure visualizer
    Visualizer.pif_indices = args.debug_cif_indices
    Visualizer.paf_indices = args.debug_caf_indices
    Visualizer.dpaf_indices = args.debug_dcaf_indices
    if args.debug_cif_indices or args.debug_caf_indices or args.debug_dcaf_indices:
        args.debug = True


def factory(headnames, strides, skeleton=False):
    if isinstance(headnames[0], (list, tuple)):
        return [factory(task_headnames, task_strides)
                for task_headnames, task_strides in zip(headnames, strides)]

    encoders = [factory_head(head_name, stride)
                for head_name, stride in zip(headnames, strides)]
    if skeleton:
        encoders.append(Skeleton())

    return encoders


def factory_head(head_name, stride):
    cif_m = re.match('[cp]if([0-9]*)$', head_name)
    if cif_m is not None:
        n_keypoints = int(cif_m.group(1)) if cif_m.group(1) else 17
        LOG.debug('using %d keypoints for pif', n_keypoints)

        LOG.info('selected encoder Pif for %s with %d keypoints', head_name, n_keypoints)
        return Pif(stride, n_keypoints=n_keypoints, sigmas=COCO_PERSON_SIGMAS)

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
        return Caf(stride,
                   n_keypoints=n_keypoints,
                   skeleton=skeleton,
                   sigmas=COCO_PERSON_SIGMAS,
                   sparse_skeleton=sparse_skeleton,
                   only_in_field_of_view=only_in_field_of_view)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
