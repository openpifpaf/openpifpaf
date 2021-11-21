"""Drawing primitives."""

from .animation_frame import AnimationFrame
from .canvas import Canvas, canvas, image_canvas, white_screen
from .cli import cli, configure
from .fields import arrows, boxes, boxes_wh, circles, margins, quiver
from .painters import CrowdPainter, DetectionPainter, KeypointPainter
from .annotation_painter import AnnotationPainter

__all__ = [
    'AnimationFrame',
    'Canvas',
    'arrows', 'boxes', 'boxes_wh', 'circles', 'margins', 'quiver',
    'CrowdPainter', 'DetectionPainter', 'KeypointPainter',
    'AnnotationPainter',
]
