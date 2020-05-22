import logging

from .base import BaseVisualizer
from ..annotation import AnnotationDet
from .. import show

LOG = logging.getLogger(__name__)


class CifDet(BaseVisualizer):
    show_margin = False
    show_confidences = False
    show_regressions = False
    show_background = False

    def __init__(self, head_name, *, stride=1, categories=None):
        super().__init__(head_name)

        self.stride = stride
        self.categories = categories

        self.detection_painter = show.DetectionPainter(xy_scale=stride)

    def targets(self, field, detections):
        assert self.categories is not None

        annotations = [
            AnnotationDet(self.categories).set(det[0] - 1, None, det[1])
            for det in detections
        ]

        self._confidences(field[0])
        self._regressions(field[1], field[2], annotations=annotations)

    def predicted(self, field, *, annotations=None):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 4:6],
                          annotations=annotations,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return

        for f in self.indices:
            LOG.debug('%s', self.categories[f])

            with self.image_canvas(self._processed_image) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap='Greens')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, wh_fields, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return

        for f in self.indices:
            LOG.debug('%s', self.categories[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.detection_painter.annotations(ax, annotations, color='gray')
                q = show.quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.stride, uv_is_offset=uv_is_offset,
                                cmap='Greens', clim=(0.5, 1.0), width=0.001)
                show.boxes_wh(ax, wh_fields[f, 0], wh_fields[f, 1],
                              confidence_field=confidence_field,
                              regression_field=regression_fields[f, :2],
                              xy_scale=self.stride, cmap='Greens', fill=False,
                              regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    show.margins(ax, regression_fields[f, :6], xy_scale=self.stride)

                self.colorbar(ax, q)
