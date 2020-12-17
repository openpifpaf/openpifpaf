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
    group.add_argument('--image-width', default=None, type=float,
                       help='image width for matplotlib (in inches)')
    group.add_argument('--image-height', default=None, type=float,
                       help='image height for matplotlib (in inches)')
    group.add_argument('--image-dpi-factor', default=Canvas.image_dpi_factor, type=float,
                       help='increase dpi of output image by this factor')
    group.add_argument('--image-min-dpi', default=Canvas.image_min_dpi, type=float,
                       help='minimum dpi of image output')
    group.add_argument('--show-file-extension', default=Canvas.out_file_extension,
                       help='default file extension')
    group.add_argument('--textbox-alpha', default=KeypointPainter.textbox_alpha, type=float)
    group.add_argument('--text-color', default=KeypointPainter.text_color)
    group.add_argument('--font-size', default=KeypointPainter.font_size, type=int)
    assert not KeypointPainter.monocolor_connections
    group.add_argument('--monocolor-connections', default=False, action='store_true')
    group.add_argument('--line-width', default=None, type=int)
    group.add_argument('--skeleton-solid-threshold',
                       default=KeypointPainter.solid_threshold, type=float)

    group.add_argument('--show-box', default=False, action='store_true')
    group.add_argument('--white-overlay',
                       nargs='?', default=False, const=0.8, type=float)
    group.add_argument('--show-joint-scales', default=False, action='store_true')
    group.add_argument('--show-joint-confidences', default=False, action='store_true')
    group.add_argument('--show-decoding-order', default=False, action='store_true')
    group.add_argument('--show-frontier-order', default=False, action='store_true')
    group.add_argument('--show-only-decoded-connections', default=False, action='store_true')

    group.add_argument('--video-fps', default=AnimationFrame.video_fps, type=float,
                       help='output video frame rate (frames per second)')
    group.add_argument('--video-dpi', default=AnimationFrame.video_dpi, type=float,
                       help='output video resolution (dots per inch)')


def configure(args):
    Canvas.all_images_directory = args.save_all
    Canvas.show = args.show
    if args.image_width is not None:
        Canvas.image_width = args.image_width
    if args.image_height is not None:
        Canvas.image_height = args.image_height
        if args.image_width is None:
            # if only image height is provided, do not force image width
            Canvas.image_width = None
    Canvas.image_dpi_factor = args.image_dpi_factor
    Canvas.white_overlay = args.white_overlay
    Canvas.image_min_dpi = args.image_min_dpi
    Canvas.out_file_extension = args.show_file_extension

    KeypointPainter.show_box = args.show_box
    KeypointPainter.show_joint_scales = args.show_joint_scales
    KeypointPainter.show_joint_confidences = args.show_joint_confidences
    KeypointPainter.show_decoding_order = args.show_decoding_order
    KeypointPainter.show_frontier_order = args.show_frontier_order
    KeypointPainter.show_only_decoded_connections = args.show_only_decoded_connections

    KeypointPainter.textbox_alpha = args.textbox_alpha
    KeypointPainter.text_color = args.text_color
    KeypointPainter.monocolor_connections = args.monocolor_connections
    KeypointPainter.line_width = args.line_width
    KeypointPainter.solid_threshold = args.skeleton_solid_threshold
    KeypointPainter.font_size = args.font_size

    AnimationFrame.video_fps = args.video_fps
    AnimationFrame.video_dpi = args.video_dpi
    AnimationFrame.show = args.show
