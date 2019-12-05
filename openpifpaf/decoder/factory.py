import logging

from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_CONNECTIONS
from .pif import Pif
from .pif_hr import PifHr
from .pifpaf import PifPaf
from .pifpaf_dijkstra import PifPafDijkstra
from .processor import Processor
from .visualizer import Visualizer

LOG = logging.getLogger(__name__)


def cli(parser, *,
        force_complete_pose=True,
        seed_threshold=0.2,
        instance_threshold=0.0,
        keypoint_threshold=None,
        workers=None):
    group = parser.add_argument_group('decoder configuration')
    group.add_argument('--seed-threshold', default=seed_threshold, type=float,
                       help='minimum threshold for seeds')
    group.add_argument('--instance-threshold', type=float,
                       default=instance_threshold,
                       help='filter instances by score')
    group.add_argument('--keypoint-threshold', type=float,
                       default=keypoint_threshold,
                       help='filter keypoints by score')
    group.add_argument('--decoder-workers', default=workers, type=int,
                       help='number of workers for pose decoding')
    group.add_argument('--experimental-decoder', default=False, action='store_true',
                       help='use an experimental decoder')
    group.add_argument('--extra-coupling', default=0.0, type=float,
                       help='extra coupling')

    if force_complete_pose:
        group.add_argument('--no-force-complete-pose', dest='force_complete_pose',
                           default=True, action='store_false')
    else:
        group.add_argument('--force-complete-pose', dest='force_complete_pose',
                           default=False, action='store_true')

    group.add_argument('--debug-pif-indices', default=[], nargs='+',
                       help=('indices of PIF fields to create debug plots for '
                             '(group with comma, e.g. "0,1 2" to create one plot '
                             'with field 0 and 1 and another plot with field 2)'))
    group.add_argument('--debug-paf-indices', default=[], nargs='+',
                       help=('indices of PAF fields to create debug plots for '
                             '(same grouping behavior as debug-pif-indices)'))
    group.add_argument('--debug-file-prefix', default=None,
                       help='save debug plots with this prefix')
    group.add_argument('--profile-decoder', default=None, action='store_true',
                       help='profile decoder')

    group = parser.add_argument_group('PifPaf decoders')
    assert PifPaf.fixed_b == PifPafDijkstra.fixed_b
    group.add_argument('--fixed-b', default=PifPaf.fixed_b, type=float,
                       help='overwrite b with fixed value, e.g. 0.5')
    assert PifPaf.pif_fixed_scale == PifPafDijkstra.pif_fixed_scale
    group.add_argument('--pif-fixed-scale', default=PifPaf.pif_fixed_scale, type=float,
                       help='overwrite pif scale with a fixed value')
    group.add_argument('--pif-th', default=PifHr.v_threshold, type=float,
                       help='pif threshold')
    assert PifPaf.paf_th == PifPafDijkstra.paf_th
    group.add_argument('--paf-th', default=PifPaf.paf_th, type=float,
                       help='paf threshold')
    assert PifPaf.connection_method == PifPafDijkstra.connection_method
    group.add_argument('--connection-method',
                       default=PifPaf.connection_method,
                       choices=('median', 'max', 'blend'),
                       help='connection method to use, max is faster')


def factory_from_args(args, model, device=None):
    # configure PifPaf
    PifPaf.fixed_b = args.fixed_b
    PifPaf.pif_fixed_scale = args.pif_fixed_scale
    PifPaf.paf_th = args.paf_th
    PifPaf.connection_method = args.connection_method
    PifPaf.force_complete = args.force_complete_pose

    # configure PifPafDijkstra
    PifPafDijkstra.fixed_b = args.fixed_b
    PifPafDijkstra.pif_fixed_scale = args.pif_fixed_scale
    PifPafDijkstra.paf_th = args.paf_th
    PifPafDijkstra.connection_method = args.connection_method
    PifPafDijkstra.force_complete = args.force_complete_pose

    # configure Pif
    Pif.pif_fixed_scale = args.pif_fixed_scale

    # configure PifHr
    PifHr.v_threshold = args.pif_th

    debug_visualizer = None
    if args.debug_pif_indices or args.debug_paf_indices:
        debug_visualizer = Visualizer(
            args.debug_pif_indices, args.debug_paf_indices,
            file_prefix=args.debug_file_prefix,
            skeleton=COCO_PERSON_SKELETON + DENSER_COCO_PERSON_CONNECTIONS,
        )

    # default value for keypoint filter depends on whether complete pose is forced
    if args.keypoint_threshold is None:
        args.keypoint_threshold = 0.001 if not args.force_complete_pose else 0.0

    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       debug_visualizer is None:
        args.decoder_workers = args.batch_size

    decode = factory_decode(model,
                            experimental=args.experimental_decoder,
                            seed_threshold=args.seed_threshold,
                            extra_coupling=args.extra_coupling,
                            multi_scale=args.multi_scale,
                            multi_scale_hflip=args.multi_scale_hflip,
                            debug_visualizer=debug_visualizer)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     debug_visualizer=debug_visualizer,
                     profile=args.profile_decoder,
                     worker_pool=args.decoder_workers,
                     device=device)


def factory_decode(model, *,
                   extra_coupling=0.0,
                   experimental=False,
                   multi_scale=False,
                   multi_scale_hflip=True,
                   **kwargs):
    """Instantiate a decoder."""

    head_names = (
        tuple(model.head_names)
        if hasattr(model, 'head_names')
        else tuple(h.shortname for h in model.head_nets)
    )
    LOG.debug('head names = %s', head_names)

    if head_names in (('pif',),):
        return Pif(model.head_strides[-1], head_index=0, **kwargs)

    if head_names in (('pif', 'paf'),
                      ('pif', 'paf44'),
                      ('pif', 'paf16'),
                      ('pif', 'wpaf')):
        return PifPaf(model.head_strides[-1],
                      keypoints=COCO_KEYPOINTS,
                      skeleton=COCO_PERSON_SKELETON,
                      **kwargs)

    if head_names in (('pif', 'paf', 'paf25'),):
        stride = model.head_strides[-1]
        pif_index = 0
        paf_index = 1
        pif_min_scale = 0.0
        paf_min_distance = 0.0
        paf_max_distance = None
        if multi_scale and multi_scale_hflip:
            resolutions = [1, 1.5, 2, 3, 5] * 2
            stride = [model.head_strides[-1] * r for r in resolutions]
            if not experimental:
                pif_index = [v * 3 for v in range(10)]
                paf_index = [v * 3 + 1 for v in range(10)]
            else:
                pif_index = [v * 2 for v in range(10)]
                paf_index = [v * 2 + 1 for v in range(10)]
            pif_min_scale = [0.0, 12.0, 16.0, 24.0, 40.0] * 2
            paf_min_distance = [v * 3.0 for v in pif_min_scale]
            paf_max_distance = [160.0, 240.0, 320.0, 480.0, None] * 2
            # paf_max_distance = [128.0, 192.0, 256.0, 384.0, None] * 2
        elif multi_scale and not multi_scale_hflip:
            resolutions = [1, 1.5, 2, 3, 5]
            stride = [model.head_strides[-1] * r for r in resolutions]
            if not experimental:
                pif_index = [v * 3 for v in range(5)]
                paf_index = [v * 3 + 1 for v in range(5)]
            else:
                pif_index = [v * 2 for v in range(5)]
                paf_index = [v * 2 + 1 for v in range(5)]
            pif_min_scale = [0.0, 12.0, 16.0, 24.0, 40.0]
            paf_min_distance = [v * 3.0 for v in pif_min_scale]
            paf_max_distance = [160.0, 240.0, 320.0, 480.0, None]
            # paf_max_distance = [128.0, 192.0, 256.0, 384.0, None]

        if experimental:
            LOG.warning('using experimental decoder')
            confidence_scales = (
                [1.0 for _ in COCO_PERSON_SKELETON] +
                [extra_coupling for _ in DENSER_COCO_PERSON_CONNECTIONS]
            )
            return PifPafDijkstra(
                stride,
                pif_index=pif_index,
                paf_index=paf_index,
                pif_min_scale=pif_min_scale,
                paf_min_distance=paf_min_distance,
                paf_max_distance=paf_max_distance,
                keypoints=COCO_KEYPOINTS,
                skeleton=COCO_PERSON_SKELETON + DENSER_COCO_PERSON_CONNECTIONS,
                confidence_scales=confidence_scales,
                **kwargs
            )

        return PifPaf(
            stride,
            pif_index=pif_index,
            paf_index=paf_index,
            pif_min_scale=pif_min_scale,
            paf_min_distance=paf_min_distance,
            paf_max_distance=paf_max_distance,
            keypoints=COCO_KEYPOINTS,
            skeleton=COCO_PERSON_SKELETON,
            **kwargs
        )

    raise Exception('decoder unknown for head names: {}'.format(head_names))
