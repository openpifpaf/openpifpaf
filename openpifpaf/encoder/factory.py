import logging
import re

from .paf import Paf
from .pif import Pif
from .skeleton import Skeleton

from ..data import (COCO_PERSON_SKELETON, COCO_PERSON_SIGMAS, DENSER_COCO_PERSON_SKELETON,
                    KINEMATIC_TREE_SKELETON, DENSER_COCO_PERSON_CONNECTIONS)

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('pif encoder')
    group.add_argument('--pif-side-length', default=Pif.side_length, type=int,
                       help='side length of the PIF field')

    group = parser.add_argument_group('paf encoder')
    group.add_argument('--paf-min-size', default=Paf.min_size, type=int,
                       help='min side length of the PAF field')
    group.add_argument('--paf-fixed-size', default=Paf.fixed_size, action='store_true',
                       help='fixed paf size')
    group.add_argument('--paf-aspect-ratio', default=Paf.aspect_ratio, type=float,
                       help='paf width relative to its length')


def factory(args, strides):
    # configure Pif
    Pif.side_length = args.pif_side_length

    # configure Paf
    Paf.min_size = args.paf_min_size
    Paf.fixed_size = args.paf_fixed_size
    Paf.aspect_ratio = args.paf_aspect_ratio

    return factory_heads(args.headnets, strides)


def factory_heads(headnames, strides):
    if isinstance(headnames[0], (list, tuple)):
        return [factory_heads(task_headnames, task_strides)
                for task_headnames, task_strides in zip(headnames, strides)]

    encoders = [factory_head(head_name, stride)
                for head_name, stride in zip(headnames, strides)]
    if headnames[-1] == 'skeleton' and len(headnames) == len(strides) + 1:
        encoders.append(Skeleton())

    return encoders


def factory_head(head_name, stride):
    if head_name in ('pif',
                     'ppif',
                     'pifb',
                     'pifs') or \
       re.match('pif([0-9]+)$', head_name) is not None:

        m = re.match('pif([0-9]+)$', head_name)
        if m is not None:
            n_keypoints = int(m.group(1))
            LOG.debug('using %d keypoints for pif', n_keypoints)
        else:
            n_keypoints = 17

        LOG.info('selected encoder Pif for %s with %d keypoints', head_name, n_keypoints)
        return Pif(stride, n_keypoints=n_keypoints, sigmas=COCO_PERSON_SIGMAS)

    if head_name in ('paf',
                     'pafs',
                     'wpaf',
                     'pafb') or \
       re.match('paf[s]?([0-9]+)$', head_name) is not None:
        if head_name in ('paf', 'paf19', 'pafs', 'wpaf', 'pafb'):
            n_keypoints = 17
            skeleton = COCO_PERSON_SKELETON
        elif head_name in ('paf16',):
            n_keypoints = 17
            skeleton = KINEMATIC_TREE_SKELETON
        elif head_name in ('paf44',):
            n_keypoints = 17
            skeleton = DENSER_COCO_PERSON_SKELETON
        elif head_name in ('paf25', 'pafs25'):
            n_keypoints = 17
            skeleton = DENSER_COCO_PERSON_CONNECTIONS
        else:
            raise Exception('unknown skeleton type of head')

        LOG.info('selected encoder Paf for %s', head_name)
        return Paf(stride, n_keypoints=n_keypoints, skeleton=skeleton, sigmas=COCO_PERSON_SIGMAS)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
