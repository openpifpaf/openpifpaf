import logging

from .animation_frame import AnimationFrame
from .canvas import Canvas
from .painters import KeypointPainter

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('show')
    group.add_argument('--save-all', nargs='?', default=None, const='all-images/',
                       help='every plot is saved (optional to specify directory)')
    assert not Canvas.show
    assert not AnimationFrame.show
    group.add_argument('--show', default=False, action='store_true',
                       help='show every plot, i.e., call matplotlib show()')
    group.add_argument('--figure-width', default=Canvas.figure_width, type=float,
                       help='figure width for matplotlib (in inches)')
    group.add_argument('--image-dpi-factor', default=Canvas.image_dpi_factor, type=float,
                       help='increase dpi of output image by this factor')

    group.add_argument('--show-box', default=False, action='store_true')
    group.add_argument('--white-overlay',
                       nargs='?', default=False, const=0.8, type=float)
    group.add_argument('--show-joint-scales', default=False, action='store_true')
    group.add_argument('--show-joint-confidences', default=False, action='store_true')
    group.add_argument('--show-decoding-order', default=False, action='store_true')
    group.add_argument('--show-frontier-order', default=False, action='store_true')
    group.add_argument('--show-only-decoded-connections', default=False, action='store_true')

    group.add_argument('--video-fps', default=AnimationFrame.video_fps, type=float)
    group.add_argument('--video-dpi', default=AnimationFrame.video_dpi, type=float)


def configure(args):
    Canvas.all_images_directory = args.save_all
    Canvas.show = args.show
    Canvas.figure_width = args.figure_width
    Canvas.image_dpi_factor = args.image_dpi_factor
    Canvas.white_overlay = args.white_overlay

    KeypointPainter.show_box = args.show_box
    KeypointPainter.show_joint_scales = args.show_joint_scales
    KeypointPainter.show_joint_confidences = args.show_joint_confidences
    KeypointPainter.show_decoding_order = args.show_decoding_order
    KeypointPainter.show_frontier_order = args.show_frontier_order
    KeypointPainter.show_only_decoded_connections = args.show_only_decoded_connections

    AnimationFrame.video_fps = args.video_fps
    AnimationFrame.video_dpi = args.video_dpi
    AnimationFrame.show = args.show
