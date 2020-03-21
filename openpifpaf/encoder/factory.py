import logging
import re

from .caf import Caf
from .cif import Cif
from .visualizer import CifVisualizer, CafVisualizer

from ..data import (COCO_KEYPOINTS, COCO_PERSON_SKELETON, COCO_PERSON_SIGMAS,
                    DENSER_COCO_PERSON_SKELETON,
                    KINEMATIC_TREE_SKELETON, DENSER_COCO_PERSON_CONNECTIONS)

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

    group = parser.add_argument_group('debug')
    group.add_argument('--debug-indices', default=[], nargs='+',
                       help='indices of fields to create debug plots for of the form cif:5')


def configure(args):
    # configure CIF
    Cif.side_length = args.cif_side_length

    # configure CAF
    Caf.min_size = args.caf_min_size
    Caf.fixed_size = args.caf_fixed_size
    Caf.aspect_ratio = args.caf_aspect_ratio

    # configure visualizer
    if args.debug_indices:
        args.debug = True
    args.debug_indices = [di.partition(':') for di in args.debug_indices]
    args.debug_indices = [(di[0], int(di[2])) for di in args.debug_indices]


def factory(headnames, strides, debug_indices):
    if isinstance(headnames[0], (list, tuple)):
        return [factory(task_headnames, task_strides, debug_indices)
                for task_headnames, task_strides in zip(headnames, strides)]

    debug_indices = [
        [f for dhi, f in debug_indices if dhi == head_name]
        for head_name in headnames
    ]

    encoders = [factory_head(head_name, stride, di)
                for head_name, stride, di in zip(headnames, strides, debug_indices)]
    return encoders


def factory_head(head_name, stride, debug_indices):
    cif_m = re.match('[cp]if([0-9]*)$', head_name)
    if cif_m is not None:
        n_keypoints = int(cif_m.group(1)) if cif_m.group(1) else 17
        LOG.debug('using %d keypoints for CIF', n_keypoints)

        LOG.info('selected encoder CIF for %s with %d keypoints', head_name, n_keypoints)
        visualizer = CifVisualizer(head_name, stride, debug_indices,
                                   keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
        return Cif(stride,
                   n_keypoints=n_keypoints,
                   sigmas=COCO_PERSON_SIGMAS,
                   visualizer=visualizer)

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
        visualizer = CafVisualizer(head_name, stride, debug_indices,
                                   keypoints=COCO_KEYPOINTS, skeleton=skeleton)
        return Caf(stride,
                   n_keypoints=n_keypoints,
                   skeleton=skeleton,
                   sigmas=COCO_PERSON_SIGMAS,
                   sparse_skeleton=sparse_skeleton,
                   only_in_field_of_view=only_in_field_of_view,
                   visualizer=visualizer)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
