import logging

from .base import BaseVisualizer

LOG = logging.getLogger(__name__)


class CifHr(BaseVisualizer):
    show = False

    def __init__(self, *, stride=1, field_names=None):
        super().__init__(('cif', 'cifdet'))

        self.stride = stride
        self.field_names = field_names

    def predicted(self, fields):
        if not self.show:
            return

        for f in self.indices:
            LOG.debug('%d (field name: %s)',
                      f, self.field_names[f] if self.field_names else 'unknown')
            with self.image_canvas(self._processed_image) as ax:
                o = ax.imshow(fields[f], alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')
                self.colorbar(ax, o)
