from collections import defaultdict
import logging
from typing import Optional

from .cifcaf import CifCaf
from .cifdet import CifDet
from .decoder import Decoder
from .multi import Multi
from . import utils
from .profiler import Profiler
from .profiler_autograd import ProfilerAutograd

LOG = logging.getLogger(__name__)

DECODERS = {CifDet, CifCaf}


def cli(parser, *, workers=None):
    group = parser.add_argument_group('decoder configuration')

    available_decoders = [dec.__name__.lower() for dec in DECODERS]
    group.add_argument('--decoder', default=None, nargs='+',
                       help='Decoders to be considered: {}.'.format(available_decoders))

    group.add_argument('--seed-threshold', default=utils.CifSeeds.threshold, type=float,
                       help='minimum threshold for seeds')
    assert utils.nms.Detection.instance_threshold == utils.nms.Keypoints.instance_threshold
    group.add_argument('--instance-threshold', type=float, default=None,
                       help=('filter instances by score (0.0 with --force-complete-pose '
                             'else {})'.format(utils.nms.Keypoints.instance_threshold)))
    group.add_argument('--decoder-workers', default=workers, type=int,
                       help='number of workers for pose decoding')
    group.add_argument('--caf-seeds', default=False, action='store_true',
                       help='[experimental]')

    group.add_argument('--profile-decoder', nargs='?', const='profile_decoder.prof', default=None,
                       help='specify out .prof file or nothing for default file name')

    group = parser.add_argument_group('CifCaf decoders')
    group.add_argument('--cif-th', default=utils.CifHr.v_threshold, type=float,
                       help='cif threshold')
    group.add_argument('--caf-th', default=utils.CafScored.default_score_th, type=float,
                       help='caf threshold')

    for dec in DECODERS:
        dec.cli(parser)


def configure(args):
    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       not args.debug:
        args.decoder_workers = args.batch_size
    if args.instance_threshold is None:
        if args.force_complete_pose:
            args.instance_threshold = 0.0
        else:
            args.instance_threshold = utils.nms.Keypoints.instance_threshold

    # configure Factory
    Factory.decoder_filter_from_args(args.decoder)
    Factory.profile = args.profile_decoder
    Factory.profile_device = args.device

    # configure CifHr
    utils.CifHr.v_threshold = args.cif_th

    # configure CifSeeds
    utils.CifSeeds.threshold = args.seed_threshold

    # configure CafScored
    utils.CafScored.default_score_th = args.caf_th

    # configure generators
    Decoder.default_worker_pool = args.decoder_workers

    # configure nms
    utils.nms.Detection.instance_threshold = args.instance_threshold
    utils.nms.Keypoints.instance_threshold = args.instance_threshold

    # TODO: caf seeds
    assert not args.caf_seeds, 'not implemented'

    for dec in DECODERS:
        dec.configure(args)


class Factory:
    decoder_filter: Optional[dict] = None
    profile = False
    profile_device = None

    @classmethod
    def decoder_filter_from_args(cls, list_str):
        if list_str is None:
            cls.decoder_filter = None
            return

        cls.decoder_filter = defaultdict(list)
        for dec_str in list_str:
            # pylint: disable=unsupported-assignment-operation,unsupported-membership-test,unsubscriptable-object
            if ':' not in dec_str:
                if dec_str not in cls.decoder_filter:
                    cls.decoder_filter[dec_str] = []
                continue

            dec_str, _, index = dec_str.partition(':')
            index = int(index)
            cls.decoder_filter[dec_str].append(index)

        LOG.debug('setup decoder filter: %s', cls.decoder_filter)

    @classmethod
    def decoders(cls, head_metas):
        if cls.decoder_filter is not None:
            # pylint: disable=unsupported-membership-test,unsubscriptable-object
            decoders_by_class = {dec_class.__name__.lower(): dec_class
                                 for dec_class in DECODERS}
            decoders_by_class = {c: ds.factory(head_metas)
                                 for c, ds in decoders_by_class.items()
                                 if c in cls.decoder_filter}
            decoders_by_class = {c: [d
                                     for i, d in enumerate(ds)
                                     if (not cls.decoder_filter[c]
                                         or i in cls.decoder_filter[c])]
                                 for c, ds in decoders_by_class.items()}
            decoders = [d for ds in decoders_by_class.values() for d in ds]
            LOG.debug('filtered to %d decoders', len(decoders))
        else:
            decoders = [
                d
                for dec_class in DECODERS
                for d in dec_class.factory(head_metas)
            ]
            LOG.debug('created %d decoders', len(decoders))

        if not decoders:
            LOG.warning('no decoders found for heads %s', [meta.name for meta in head_metas])

        return decoders

    @classmethod
    def __call__(cls, head_metas):
        """Instantiate decoders."""
        # TODO implement!
        # dense_coupling=args.dense_coupling,
        # dense_connections=args.dense_connections,

        LOG.debug('head names = %s', [meta.name for meta in head_metas])
        decoders = cls.decoders(head_metas)

        if cls.profile:
            decode = decoders[0]
            decode.__class__.__call__ = Profiler(
                decode.__call__, out_name=cls.profile)
            decode.fields_batch = ProfilerAutograd(
                decode.fields_batch, device=cls.profile_device, out_name=cls.profile)

        return Multi(decoders)

        # TODO implement!
        # skeleton = head_nets[1].meta.skeleton
        # if dense_connections:
        #     field_config.confidence_scales = (
        #         [1.0 for _ in skeleton] +
        #         [dense_coupling for _ in head_nets[2].meta.skeleton]
        #     )
        #     skeleton = skeleton + head_nets[2].meta.skeleton


factory = Factory.__call__
