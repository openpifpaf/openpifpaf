import logging

from .base import BaseVisualizer

LOG = logging.getLogger(__name__)


class Occupancy(BaseVisualizer):
    show = False

    def __init__(self, *, field_names=None):
        super().__init__('occupancy')
        self.field_names = field_names

    def predicted(self, occupancy):
        if not self.show:
            return

        for f in self.indices:
            LOG.debug('%d (field name: %s)',
                      f, self.field_names[f] if self.field_names else 'unknown')
            with self.image_canvas(self.image) as ax:
                occ = occupancy.occupied[f].copy()
                occ[occ > 0] = 1.0
                reduced_image = self._processed_image[::occupancy.reduction, ::occupancy.reduction]
                with self.image_canvas(reduced_image) as ax:
                    ax.imshow(occ, alpha=0.5)
