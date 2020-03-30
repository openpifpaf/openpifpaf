import logging

from .base import BaseVisualizer
from .. import show

LOG = logging.getLogger(__name__)


class Seeds(BaseVisualizer):
    show = False
    show_confidences = False

    def __init__(self, *, stride=1):
        super().__init__('seeds')
        self.stride = stride

    def predicted(self, seeds):
        if not self.show:
            return

        field_indices = {f for _, f, __, ___, ____ in seeds}

        with self.image_canvas(self._processed_image) as ax:
            show.white_screen(ax)
            for f in field_indices:
                x = [xx * self.stride for _, ff, xx, __, ___ in seeds if ff == f]
                y = [yy * self.stride for _, ff, __, yy, ___ in seeds if ff == f]
                c = [cc for cc, ff, _, __, ___ in seeds if ff == f]
                ax.plot(x, y, 'o')
                if self.show_confidences:
                    for xx, yy, cc in zip(x, y, c):
                        ax.text(xx, yy, '{:.2f}'.format(cc))
