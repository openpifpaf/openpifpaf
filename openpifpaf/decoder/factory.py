import logging

from ..data import COCO_PERSON_SKELETON, DENSER_COCO_PERSON_CONNECTIONS
from .decoder import Decoder
from .paf_stack import PafStack
from .pif import Pif
from .pifpaf import PifPaf
from .pifpaf_dijkstra import PifPafDijkstra
from .processor import Processor
from .visualizer import Visualizer


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
    group.add_argument('--fixed-b', default=PifPaf.fixed_b, type=float,
                       help='overwrite b with fixed value, e.g. 0.5')
    group.add_argument('--pif-fixed-scale', default=PifPaf.pif_fixed_scale, type=float,
                       help='overwrite pif scale with a fixed value')
    group.add_argument('--paf-th', default=PifPaf.paf_th, type=float,
                       help='paf threshold')
    group.add_argument('--connection-method',
                       default=PifPaf.connection_method,
                       choices=('median', 'max', 'blend'),
                       help='connection method to use, max is faster')

    for decoder in Decoder.__subclasses__():
        decoder.cli(parser)


def factory_from_args(args, model, device=None):
    for decoder in Decoder.__subclasses__():
        decoder.apply_args(args)

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
                            debug_visualizer=debug_visualizer)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     debug_visualizer=debug_visualizer,
                     profile=args.profile_decoder,
                     worker_pool=args.decoder_workers,
                     device=device)


def factory_decode(model, *, extra_coupling=0.0, experimental=False, **kwargs):
    """Instantiate a decoder for the given model.

    All subclasses of decoder.Decoder are checked for a match.
    """
    head_names = tuple(h.shortname for h in model.head_nets)

    if head_names in (('pif',),):
        return Pif(model.io_scales()[-1],
                   head_index=0,
                   **kwargs)

    if head_names in (('pif', 'paf'),
                      ('pif', 'paf44'),
                      ('pif', 'paf16'),
                      ('pif', 'wpaf')):
        return PifPaf(model.io_scales()[-1],
                      skeleton=COCO_PERSON_SKELETON,
                      **kwargs)

    if head_names in (('pif', 'paf', 'paf25'),):
        if experimental:
            logging.warning('using experimental decoder')
            confidence_scales = (
                [1.0 for _ in COCO_PERSON_SKELETON] +
                [extra_coupling for _ in DENSER_COCO_PERSON_CONNECTIONS]
            )
            return PafStack(
                (1, 2),
                PifPafDijkstra(model.io_scales()[-1],
                               skeleton=COCO_PERSON_SKELETON + DENSER_COCO_PERSON_CONNECTIONS,
                               confidence_scales=confidence_scales,
                               **kwargs),
            )

        return PifPaf(model.io_scales()[-1],
                      skeleton=COCO_PERSON_SKELETON,
                      **kwargs)

    raise Exception('unknown head nets {} for decoder'.format(head_names))
