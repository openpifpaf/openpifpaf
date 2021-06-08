import logging
from openpifpaf.encoder.annrescaler_cent import AnnRescalerCent
from openpifpaf.encoder.cif_cent import CifCent

from .annrescaler import AnnRescaler, AnnRescalerDet
from .annrescaler_ball import AnnRescalerBall
from .caf import Caf
from .cif import Cif
from .cif_ball import CifBall
from .cifdet import CifDet
from .seg import Seg
from .pan import PanopticTargetGenerator
from .. import network, visualizer
from ..datasets.constants import _COCO_PANOPTIC_THING_LIST
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
    if isinstance(meta, network.heads.PanopticDeeplabMeta):
        LOG.info('selected encoder PAN')
        
        coco_panoptic_thing_list = _COCO_PANOPTIC_THING_LIST
        if meta.num_classes[0] == 2:
            coco_panoptic_thing_list = [_COCO_PANOPTIC_THING_LIST[0]]
        # coco_panoptic_thing_list = [_COCO_PANOPTIC_THING_LIST[0]]       # to avoid having ball in semantic mask
        print('Things in Panoptic head:', coco_panoptic_thing_list)

        return PanopticTargetGenerator(coco_panoptic_thing_list,
                        sigma=8, ignore_stuff_in_offset=True,
                        small_instance_area=0,
                        small_instance_weight=1,
                        ignore_crowd_in_semantic=True)
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
        # # print('!!!!!!!!!!!!!',meta.name)
        # ball = False
        # if meta.name in ['cifcentball', 'cifball']:
        #     ball = True
        # # print('!!!!!!!!!!!!!',ball)
        if meta.name == 'ball':
            print('!!!!!!! BALL !!!!!!')
            return CifBall(AnnRescalerBall(stride, len(meta.keypoints), meta.pose),
                    name=meta.name,
                    sigmas=meta.sigmas,
                    visualizer=vis)

        elif meta.name == 'cent':
            print('!!!!!!! CENTER !!!!!!')
            return CifCent(AnnRescalerCent(stride, len(meta.keypoints), meta.pose), 
                        name=meta.name,
                    sigmas=meta.sigmas,
                    visualizer=vis)

        return Cif(AnnRescaler(stride, len(meta.keypoints), meta.pose), 
                name=meta.name,
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

    

    ### AMA
    if isinstance(meta, network.heads.SegmentationMeta):
        LOG.info('selected encoder SEG for %s', meta.name)

        return Seg()

    raise Exception('unknown head to create an encoder: {}'.format(meta.name))
