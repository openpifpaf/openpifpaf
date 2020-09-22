import logging

from .caf_scored import CafScored
from .cif_hr import CifHr
from .cif_seeds import CifSeeds
from . import generator, nms
from .profiler import Profiler
from .profiler_autograd import ProfilerAutograd

LOG = logging.getLogger(__name__)

DECODERS = {generator.CifDet, generator.CifCaf}


def cli(parser, *, workers=None):
    group = parser.add_argument_group('decoder configuration')

    available_decoders = [dec.__name__.lower() for dec in DECODERS]
    group.add_argument('--decoder', default=None, nargs='+', choices=available_decoders,
                       help='Decoders to be considered.')

    group.add_argument('--seed-threshold', default=CifSeeds.threshold, type=float,
                       help='minimum threshold for seeds')
    assert nms.Detection.instance_threshold == nms.Keypoints.instance_threshold
    group.add_argument('--instance-threshold', type=float,
                       default=nms.Keypoints.instance_threshold,
                       help='filter instances by score')
    group.add_argument('--decoder-workers', default=workers, type=int,
                       help='number of workers for pose decoding')
    group.add_argument('--caf-seeds', default=False, action='store_true',
                       help='[experimental]')

    group.add_argument('--profile-decoder', nargs='?', const='profile_decoder.prof', default=None,
                       help='specify out .prof file or nothing for default file name')

    group = parser.add_argument_group('CifCaf decoders')
    group.add_argument('--cif-th', default=CifHr.v_threshold, type=float,
                       help='cif threshold')
    group.add_argument('--caf-th', default=CafScored.default_score_th, type=float,
                       help='caf threshold')

    for dec in DECODERS:
        dec.cli(parser)


def configure(args):
    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       not args.debug:
        args.decoder_workers = args.batch_size

    # filter decoders
    if args.decoder is not None:
        args.decoder = [dec.lower() for dec in args.decoder]
        decoders_to_remove = []
        for dec in DECODERS:
            if dec.__name__.lower() in args.decoder:
                continue
            decoders_to_remove.append(dec)
        for dec in decoders_to_remove:
            LOG.debug('removing %s from consideration', dec.__name__)
            DECODERS.remove(dec)

    # configure CifHr
    CifHr.v_threshold = args.cif_th

    # configure CifSeeds
    CifSeeds.threshold = args.seed_threshold

    # configure CafScored
    CafScored.default_score_th = args.caf_th

    # configure generators
    generator.Generator.default_worker_pool = args.decoder_workers

    # configure nms
    nms.Detection.instance_threshold = args.instance_threshold
    nms.Keypoints.instance_threshold = args.instance_threshold
    nms.Keypoints.keypoint_threshold = (args.keypoint_threshold
                                        if not args.force_complete_pose else 0.0)

    # TODO: caf seeds
    assert not args.caf_seeds, 'not implemented'

    for dec in DECODERS:
        dec.configure(args)


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

    return generator.Multi(decoders)

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
