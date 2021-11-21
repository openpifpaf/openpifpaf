import logging

from .base import Base

LOG = logging.getLogger(__name__)


class CifHr(Base):
    """Visualize the CifHr map."""

    def __init__(self, *, stride=1, field_names=None):
        super().__init__('cif')

        self.stride = stride
        self.field_names = field_names

    def predicted(self, fields, low=0.0):
        for f in self.indices('hr'):
            LOG.debug('%d (field name: %s)',
                      f, self.field_names[f] if self.field_names else 'unknown')
            with self.image_canvas(self.processed_image(), margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                o = ax.imshow(fields[f], alpha=0.9, vmin=low, vmax=low + 1.0, cmap='Oranges')
                self.colorbar(ax, o)
