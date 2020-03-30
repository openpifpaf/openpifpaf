import logging

from .base import BaseVisualizer

LOG = logging.getLogger(__name__)


class CifHr(BaseVisualizer):
    show = False

    def __init__(self, *, keypoints, stride=1):
        super().__init__('cif')

        self.keypoints = keypoints
        self.stride = stride

    def predicted(self, fields):
        if not self.show:
            return

        for f in self.indices:
            LOG.debug('%s', self.keypoints[f])
            with self.image_canvas(self.image) as ax:
                o = ax.imshow(fields[f], alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')
                self.colorbar(ax, o)
