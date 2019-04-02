import cProfile

from ..data import KINEMATIC_TREE_SKELETON, DENSER_COCO_PERSON_SKELETON
from .pifpaf import PifPaf
from .pifspafs import PifsPafs
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
    group.add_argument('--profile-decoder', default=False, action='store_true',
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
                            connection_method=args.connection_method)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     debug_visualizer=debug_visualizer)


def factory_decode(model, *,
                   seed_threshold=0.2,
                   fixed_b=None,
                   pif_fixed_scale=None,
                   debug_visualizer=None,
                   profile_decoder=False,
                   force_complete_pose=False,
                   connection_method='max'):
    headnames = tuple(h.shortname for h in model.head_nets)

    if headnames == ('pif17', 'paf19'):
        decode = PifPaf(model.io_scales()[-1], seed_threshold,
                        force_complete=force_complete_pose,
                        connection_method=connection_method,
                        debug_visualizer=debug_visualizer)
    elif headnames in (('pif', 'paf'), ('pif', 'wpaf')):
        decode = PifPaf(model.io_scales()[-1], seed_threshold,
                        force_complete=force_complete_pose,
                        connection_method=connection_method,
                        debug_visualizer=debug_visualizer,
                        fixed_b=fixed_b,
                        pif_fixed_scale=pif_fixed_scale)
    elif headnames in (('pifs17', 'pafs19'), ('pifs17', 'pafs19n2')):
        decode = PifsPafs(model.io_scales()[-1], seed_threshold,
                          force_complete=force_complete_pose,
                          connection_method=connection_method,
                          debug_visualizer=debug_visualizer,
                          pif_fixed_scale=pif_fixed_scale)
    elif headnames == ('pif17', 'pif17', 'paf19'):
        decode = PifPaf(model.io_scales()[-1], seed_threshold,
                        force_complete=force_complete_pose,
                        connection_method=connection_method,
                        debug_visualizer=debug_visualizer,
                        head_indices=(1, 2))
    elif headnames == ('paf19', 'pif17', 'paf19'):
        decode = PifPaf(model.io_scales()[-1], seed_threshold,
                        force_complete=force_complete_pose,
                        connection_method=connection_method,
                        debug_visualizer=debug_visualizer,
                        head_indices=(1, 2))
    elif headnames == ('pif17', 'paf16'):
        decode = PifPaf(model.io_scales()[-1], seed_threshold,
                        KINEMATIC_TREE_SKELETON,
                        force_complete=force_complete_pose,
                        connection_method=connection_method,
                        debug_visualizer=debug_visualizer)
    elif headnames == ('pif', 'paf44'):
        decode = PifPaf(model.io_scales()[-1], seed_threshold,
                        DENSER_COCO_PERSON_SKELETON,
                        force_complete=force_complete_pose,
                        connection_method=connection_method,
                        debug_visualizer=debug_visualizer,
                        fixed_b=fixed_b)
    else:
        raise Exception('unknown head nets {} for decoder'.format(headnames))

    if profile_decoder:
        decode.profile = cProfile.Profile()

    return decode
