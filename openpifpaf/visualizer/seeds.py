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
        """Seeds are: confidence, field_index, x, y, ..."""
        if not self.show:
            return

        field_indices = {s[1] for s in seeds}

        with self.image_canvas(self._processed_image) as ax:
            show.white_screen(ax)
            for f in field_indices:
                x = [s[2] * self.stride for s in seeds if s[1] == f]
                y = [s[3] * self.stride for s in seeds if s[1] == f]
                ax.plot(x, y, 'o')
                if self.show_confidences:
                    c = [s[0] for s in seeds if s[1] == f]
                    for xx, yy, cc in zip(x, y, c):
                        ax.text(xx, yy, '{:.2f}'.format(cc))
