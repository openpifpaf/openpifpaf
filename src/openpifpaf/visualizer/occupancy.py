import logging

from .base import Base

try:
    import scipy
except ImportError:
    scipy = None

LOG = logging.getLogger(__name__)


class Occupancy(Base):
    """Visualize occupancy map."""

    def __init__(self, *, field_names=None):
        super().__init__('occupancy')
        self.field_names = field_names

    def predicted(self, occupancy):
        for f in self.indices():
            LOG.debug('%d (field name: %s)',
                      f, self.field_names[f] if self.field_names else 'unknown')

            # occupancy maps are at a reduced scale wrt the processed image
            # pylint: disable=unsubscriptable-object
            factor = 1.0 / occupancy.reduction
            reduced_image = scipy.ndimage.zoom(self.processed_image(), (factor, factor, 1), order=1)

            with self.image_canvas(reduced_image) as ax:
                occ = occupancy.occupancy[f].copy()
                occ[occ > 0] = 1.0
                ax.imshow(occ, alpha=0.7)
