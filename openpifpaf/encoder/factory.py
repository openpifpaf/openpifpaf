import logging

from .annrescaler import AnnRescaler, AnnRescalerDet
from .caf import Caf
from .cif import Cif
from .cifdet import CifDet
from .. import network, visualizer

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


def factory(headnets, basenet_stride):
    return [factory_head(head_net, basenet_stride) for head_net in headnets]


def factory_head(head_net: network.heads.CompositeField, basenet_stride):
    meta = head_net.meta
    stride = head_net.stride(basenet_stride)

    if isinstance(meta, network.heads.DetectionMeta):
        n_categories = len(meta.categories)
        LOG.info('selected encoder CIFDET for %s with %d categories', meta.name, n_categories)
        vis = visualizer.CifDet(meta.name,
                                stride=stride,
                                categories=meta.categories)
        return CifDet(n_categories,
                      AnnRescalerDet(stride, n_categories),
                      visualizer=vis)

    if isinstance(meta, network.heads.IntensityMeta):
        LOG.info('selected encoder CIF for %s', meta.name)
        vis = visualizer.Cif(meta.name,
                             stride=stride,
                             keypoints=meta.keypoints, skeleton=meta.draw_skeleton)
        return Cif(AnnRescaler(stride, len(meta.keypoints), meta.pose),
                   sigmas=meta.sigmas,
                   visualizer=vis)

    if isinstance(meta, network.heads.AssociationMeta):
        n_keypoints = len(meta.keypoints)
        LOG.info('selected encoder CAF for %s', meta.name)
        vis = visualizer.Caf(meta.name,
                             stride=stride,
                             keypoints=meta.keypoints, skeleton=meta.skeleton)
        return Caf(AnnRescaler(stride, n_keypoints, meta.pose),
                   skeleton=meta.skeleton,
                   sigmas=meta.sigmas,
                   sparse_skeleton=meta.sparse_skeleton,
                   only_in_field_of_view=meta.only_in_field_of_view,
                   visualizer=vis)

    raise Exception('unknown head to create an encoder: {}'.format(meta.name))
