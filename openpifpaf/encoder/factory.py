import logging

from .annrescaler import AnnRescaler
from .caf import Caf
from .cif import Cif

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('encoders')
    group.add_argument('--cif-side-length', default=Cif.side_length, type=int,
                       help='side length of the CIF field')
    group.add_argument('--caf-min-size', default=Caf.min_size, type=int,
                       help='min side length of the CAF field')
    group.add_argument('--caf-fixed-size', default=Caf.fixed_size, action='store_true',
                       help='fixed caf size')
    group.add_argument('--caf-aspect-ratio', default=Caf.aspect_ratio, type=float,
                       help='CAF width relative to its length')
    assert AnnRescaler.suppress_selfhidden
    group.add_argument('--encoder-no-suppress-selfhidden',
                       dest='encoder_suppress_selfhidden',
                       default=True, action='store_false',
                       help='[experimental]')
    assert not AnnRescaler.suppress_invisible
    group.add_argument('--encoder-suppress-invisible',
                       default=False, action='store_true',
                       help='[experimental]')
    assert not AnnRescaler.suppress_collision
    group.add_argument('--encoder-suppress-collision',
                       default=False, action='store_true',
                       help='[experimental]')


def configure(args):
    # configure CIF
    Cif.side_length = args.cif_side_length

    # configure CAF
    Caf.min_size = args.caf_min_size
    Caf.fixed_size = args.caf_fixed_size
    Caf.aspect_ratio = args.caf_aspect_ratio

    # configure AnnRescaler
    AnnRescaler.suppress_selfhidden = args.encoder_suppress_selfhidden
    AnnRescaler.suppress_invisible = args.encoder_suppress_invisible
    AnnRescaler.suppress_collision = args.encoder_suppress_collision
