import logging

from .base import Base
from .. import show

LOG = logging.getLogger(__name__)


class Seeds(Base):
    """Visualize seeds."""

    def __init__(self, *, stride=1):
        super().__init__('seeds')
        self.stride = stride

    def predicted(self, seeds):
        """Seeds are: confidence, field_index, x, y, ..."""
        field_indices = self.indices()
        if not field_indices:
            return

        with self.image_canvas(self.processed_image()) as ax:
            show.white_screen(ax)
            for f in field_indices:
                if f == -1:
                    f_seeds = seeds
                else:
                    f_seeds = [s for s in seeds if f == s[1]]
                x = [s[2] * self.stride for s in f_seeds]
                y = [s[3] * self.stride for s in f_seeds]
                ax.plot(x, y, 'o')
                if f in self.indices('confidence', with_all=False):
                    c = [s[0] for s in f_seeds]
                    for xx, yy, cc in zip(x, y, c):
                        ax.text(xx, yy, '{:.2f}'.format(cc))
