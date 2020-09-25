import logging

from .animation_frame import AnimationFrame
from .canvas import SaveAll
from .painters import KeypointPainter

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('show')
    group.add_argument('--save-all', nargs='?', default=None, const='all-images/',
                       help='every plot is saved (optional to specify directory)')

    group.add_argument('--show-box', default=False, action='store_true')
    group.add_argument('--show-joint-scales', default=False, action='store_true')
    group.add_argument('--show-joint-confidences', default=False, action='store_true')
    group.add_argument('--show-decoding-order', default=False, action='store_true')
    group.add_argument('--show-frontier-order', default=False, action='store_true')
    group.add_argument('--show-only-decoded-connections', default=False, action='store_true')

    group.add_argument('--video-fps', default=AnimationFrame.video_fps, type=float)
    group.add_argument('--video-dpi', default=AnimationFrame.video_dpi, type=float)


def configure(args):
    SaveAll.directory = args.save_all

    KeypointPainter.show_box = args.show_box
    KeypointPainter.show_joint_scales = args.show_joint_scales
    KeypointPainter.show_joint_confidences = args.show_joint_confidences
    KeypointPainter.show_decoding_order = args.show_decoding_order
    KeypointPainter.show_frontier_order = args.show_frontier_order
    KeypointPainter.show_only_decoded_connections = args.show_only_decoded_connections

    AnimationFrame.video_fps = args.video_fps
    AnimationFrame.video_dpi = args.video_dpi
