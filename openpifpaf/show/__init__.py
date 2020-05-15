"""Drawing primitives."""

from .animation_frame import AnimationFrame
from .canvas import canvas, image_canvas, load_image, white_screen
from .cli import cli, configure
from .fields import arrows, boxes, boxes_wh, circles, margins, quiver
from .painters import AnnotationPainter, CrowdPainter, DetectionPainter, KeypointPainter
