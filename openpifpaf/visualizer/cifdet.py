import copy
import logging

from .base import BaseVisualizer
from ..annotation import AnnotationDet
from ..network import headmeta
from .. import show

try:
    import matplotlib.cm
    CMAP_GREENS_NAN = copy.copy(matplotlib.cm.get_cmap('Greens')).set_bad('white', alpha=0.5)
except ImportError:
    CMAP_GREENS_NAN = None

LOG = logging.getLogger(__name__)


class CifDet(BaseVisualizer):
    show_margin = False
    show_confidences = False
    show_regressions = False
    show_background = False

    def __init__(self, meta: headmeta.Detection):
        super().__init__(meta.name)
        self.meta = meta
        self.detection_painter = show.DetectionPainter()

    def targets(self, field, *, annotation_dicts):
        assert self.meta.categories is not None

        annotations = [
            AnnotationDet(self.meta.categories).set(ann['category_id'] - 1, None, ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5],
                          annotations=annotations)

    def predicted(self, field, *, annotations=None):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5],
                          annotations=annotations,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return

        for f in self.indices:
            LOG.debug('%s', self.meta.categories[f])

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_GREENS_NAN)
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, wh_fields, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return

        for f in self.indices:
            LOG.debug('%s', self.meta.categories[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.detection_painter.annotations(ax, annotations, color='gray')
                q = show.quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                cmap='Greens', clim=(0.5, 1.0), width=0.001)
                show.boxes_wh(ax, wh_fields[f, 0], wh_fields[f, 1],
                              confidence_field=confidence_field,
                              regression_field=regression_fields[f, :2],
                              xy_scale=self.meta.stride, cmap='Greens',
                              fill=False, linewidth=2,
                              regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    show.margins(ax, regression_fields[f, :6], xy_scale=self.meta.stride)

                self.colorbar(ax, q)
