import logging

from .decoder import Decoder
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

    for decoder in Decoder.__subclasses__():
        decoder.cli(parser)


def factory_from_args(args, model, device=None):
    for decoder in Decoder.__subclasses__():
        decoder.apply_args(args)

    debug_visualizer = None
    if args.debug_pif_indices or args.debug_paf_indices:
        debug_visualizer = Visualizer(args.debug_pif_indices, args.debug_paf_indices,
                                      file_prefix=args.debug_file_prefix)

    # default value for keypoint filter depends on whether complete pose is forced
    if args.keypoint_threshold is None:
        args.keypoint_threshold = 0.001 if not args.force_complete_pose else 0.0

    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       debug_visualizer is None:
        args.decoder_workers = args.batch_size

    decode = factory_decode(model,
                            seed_threshold=args.seed_threshold,
                            debug_visualizer=debug_visualizer)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     debug_visualizer=debug_visualizer,
                     profile=args.profile_decoder,
                     worker_pool=args.decoder_workers,
                     device=device)


def factory_decode(model, *, profile=None, **kwargs):
    """Instantiate a decoder for the given model.

    All subclasses of decoder.Decoder are checked for a match.
    """
    headnames = tuple(h.shortname for h in model.head_nets)

    for decoder in Decoder.__subclasses__():
        logging.debug('checking whether decoder %s matches %s',
                      decoder.__name__, headnames)
        if not decoder.match(headnames):
            continue
        logging.info('selected decoder: %s', decoder.__name__)
        return decoder(model.io_scales()[-1],
                       head_names=headnames,
                       **kwargs)

    raise Exception('unknown head nets {} for decoder'.format(headnames))
