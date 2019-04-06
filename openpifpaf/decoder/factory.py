import cProfile
import logging

from .plugin import Plugin
from .processor import Processor
from .visualizer import Visualizer


def cli(parser, force_complete_pose=True, instance_threshold=0.0):
    group = parser.add_argument_group('decoder configuration')
    group.add_argument('--seed-threshold', default=0.2, type=float,
                       help='minimum threshold for seeds')
    group.add_argument('--instance-threshold', type=float, default=instance_threshold,
                       help='filter instances by score')
    group.add_argument('--keypoint-threshold', type=float, default=None,
                       help='filter keypoints by score')

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
    group.add_argument('--connection-method',
                       default='max', choices=('median', 'max'),
                       help='connection method to use, max is faster')
    group.add_argument('--fixed-b', default=None, type=float,
                       help='overwrite b with fixed value, e.g. 0.5')
    group.add_argument('--pif-fixed-scale', default=None, type=float,
                       help='overwrite pif scale with a fixed value')
    group.add_argument('--profile-decoder', default=None, action='store_true',
                       help='profile decoder')


def factory_from_args(args, model):
    debug_visualizer = None
    if args.debug_pif_indices or args.debug_paf_indices:
        debug_visualizer = Visualizer(args.debug_pif_indices, args.debug_paf_indices)

    # default value for keypoint filter depends on whether complete pose is forced
    if args.keypoint_threshold is None:
        args.keypoint_threshold = 0.001 if not args.force_complete_pose else 0.0

    decode = factory_decode(model,
                            seed_threshold=args.seed_threshold,
                            fixed_b=args.fixed_b,
                            pif_fixed_scale=args.pif_fixed_scale,
                            profile_decoder=args.profile_decoder,
                            force_complete_pose=args.force_complete_pose,
                            connection_method=args.connection_method,
                            debug_visualizer=debug_visualizer)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     debug_visualizer=debug_visualizer)


def factory_decode(model, *,
                   profile=None,
                   **kwargs):
    headnames = tuple(h.shortname for h in model.head_nets)

    if profile is True:
        profile = cProfile.Profile()

    decode = None
    for plugin in Plugin.__subclasses__():
        logging.debug('checking whether plugin %s matches %s', plugin.__name__, headnames)
        if not plugin.match(headnames):
            continue
        logging.info('selected decoder: %s', plugin.__name__)
        return plugin(model.io_scales()[-1],
                      head_names=headnames,
                      profile=profile,
                      **kwargs)

    raise Exception('unknown head nets {} for decoder'.format(headnames))
