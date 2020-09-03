import logging

from .caf_scored import CafScored
from .cif_hr import CifHr
from .cif_seeds import CifSeeds
from . import generator, nms
from .profiler import Profiler
from .profiler_autograd import ProfilerAutograd

LOG = logging.getLogger(__name__)

DECODERS = [generator.CifDet, generator.CifCaf]


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
    group.add_argument('--caf-seeds', default=False, action='store_true',
                       help='[experimental]')

    if force_complete_pose:
        group.add_argument('--no-force-complete-pose', dest='force_complete_pose',
                           default=True, action='store_false')
    else:
        group.add_argument('--force-complete-pose', dest='force_complete_pose',
                           default=False, action='store_true')

    group.add_argument('--profile-decoder', nargs='?', const='profile_decoder.prof', default=None,
                       help='specify out .prof file or nothing for default file name')

    group = parser.add_argument_group('CifCaf decoders')
    group.add_argument('--cif-th', default=CifHr.v_threshold, type=float,
                       help='cif threshold')
    group.add_argument('--caf-th', default=CafScored.default_score_th, type=float,
                       help='caf threshold')
    group.add_argument('--connection-method',
                       default=generator.CifCaf.connection_method,
                       choices=('max', 'blend'),
                       help='connection method to use, max is faster')
    group.add_argument('--greedy', default=False, action='store_true',
                       help='greedy decoding')


def configure(args):
    # default value for keypoint filter depends on whether complete pose is forced
    if args.keypoint_threshold is None:
        args.keypoint_threshold = 0.001 if not args.force_complete_pose else 0.0

    # check consistency
    if args.force_complete_pose:
        assert args.keypoint_threshold == 0.0
    assert args.seed_threshold >= args.keypoint_threshold

    # configure CifHr
    CifHr.v_threshold = args.cif_th

    # configure CifSeeds
    CifSeeds.threshold = args.seed_threshold

    # configure CafScored
    CafScored.default_score_th = args.caf_th

    # configure generators
    generator.CifCaf.force_complete = args.force_complete_pose
    generator.CifCaf.keypoint_threshold = args.keypoint_threshold
    generator.CifCaf.greedy = args.greedy
    generator.CifCaf.connection_method = args.connection_method
    generator.Generator.default_worker_pool = args.decoder_workers

    # configure nms
    nms.Detection.instance_threshold = args.instance_threshold
    nms.Keypoints.instance_threshold = args.instance_threshold
    nms.Keypoints.keypoint_threshold = args.keypoint_threshold

    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       not args.debug:
        args.decoder_workers = args.batch_size

    # TODO: caf seeds
    assert not args.caf_seeds, 'not implemented'


def factory(head_metas, *, profile=False, profile_device=None):
    """Instantiate decoders."""
    # TODO implement!
                            # dense_coupling=args.dense_coupling,
                            # dense_connections=args.dense_connections,
                            # multi_scale=args.multi_scale,
                            # multi_scale_hflip=args.multi_scale_hflip,

    LOG.debug('head names = %s', [meta.name for meta in head_metas])
    decoders = [
        dec
        for dec_classes in DECODERS
        for dec in dec_classes.factory(head_metas)
    ]
    LOG.debug('matched %d decoders', len(decoders))
    if not decoders:
        LOG.warning('no decoders found for heads %s', [meta.name for meta in head_metas])

    if profile:
        decode = decoders[0]
        decode.__class__.__call__ = Profiler(
            decode.__call__, out_name=profile)
        decode.fields_batch = ProfilerAutograd(
            decode.fields_batch, device=profile_device, out_name=profile)

    return decoders

    # TODO implement!
        # if multi_scale:
        #     if not dense_connections:
        #         field_config.cif_indices = [v * 3 for v in range(5)]
        #         field_config.caf_indices = [v * 3 + 1 for v in range(5)]
        #     else:
        #         field_config.cif_indices = [v * 2 for v in range(5)]
        #         field_config.caf_indices = [v * 2 + 1 for v in range(5)]
        #     field_config.cif_strides = [basenet_stride / head_nets[i].meta.upsample_stride
        #                                 for i in field_config.cif_indices]
        #     field_config.caf_strides = [basenet_stride / head_nets[i].meta.upsample_stride
        #                                 for i in field_config.caf_indices]
        #     field_config.cif_min_scales = [0.0, 12.0, 16.0, 24.0, 40.0]
        #     field_config.caf_min_distances = [v * 3.0 for v in field_config.cif_min_scales]
        #     field_config.caf_max_distances = [160.0, 240.0, 320.0, 480.0, None]
        # if multi_scale and multi_scale_hflip:
        #     if not dense_connections:
        #         field_config.cif_indices = [v * 3 for v in range(10)]
        #         field_config.caf_indices = [v * 3 + 1 for v in range(10)]
        #     else:
        #         field_config.cif_indices = [v * 2 for v in range(10)]
        #         field_config.caf_indices = [v * 2 + 1 for v in range(10)]
        #     field_config.cif_strides = [basenet_stride / head_nets[i].meta.upsample_stride
        #                                 for i in field_config.cif_indices]
        #     field_config.caf_strides = [basenet_stride / head_nets[i].meta.upsample_stride
        #                                 for i in field_config.caf_indices]
        #     field_config.cif_min_scales *= 2
        #     field_config.caf_min_distances *= 2
        #     field_config.caf_max_distances *= 2

        # skeleton = head_nets[1].meta.skeleton
        # if dense_connections:
        #     field_config.confidence_scales = (
        #         [1.0 for _ in skeleton] +
        #         [dense_coupling for _ in head_nets[2].meta.skeleton]
        #     )
        #     skeleton = skeleton + head_nets[2].meta.skeleton
