import logging

import numpy as np

from .base import BaseVisualizer
from ..decoder import Annotation
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

    def targets(self, field, detections):
        assert self.categories is not None

        annotations = [
            None  # TODO
            for det in detections
        ]

        self._background(field[0])
        self._confidences(field[0])
        self._regressions(field[1], field[2], field[3], annotations)

    def predicted(self, field, *, annotations=None):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 4], annotations,
                          confidence_fields=field[:, 0], uv_is_offset=False)

    def _background(self, field):
        if not self.show_background or not self.indices:
            return

        for f in self.indices:
            with self.image_canvas(self._processed_image[::self.stride, ::self.stride]) as ax:
                ax.imshow(np.isnan(field[f]), alpha=0.9, vmin=0.0, vmax=1.0, cmap='Blues')

    def _confidences(self, confidences):
        if not self.show_confidences:
            return

        for f in self.indices:
            LOG.debug('%s', self.categories[f])

            with self.image_canvas(self._processed_image) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, w_fields, h_fields, annotations, *,
                     confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return

        for f in self.indices:
            LOG.debug('%s', self.categories[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image) as ax:
                show.white_screen(ax, alpha=0.5)
                # self.keypoint_painter.annotations(ax, annotations)
                q = show.quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.stride, uv_is_offset=uv_is_offset,
                                cmap='Oranges', clim=(0.5, 1.0), width=0.001)
                show.boxes_wh(ax, w_fields[f], h_fields[f],
                              confidence_field=confidence_field,
                              regression_field=regression_fields[f, :2],
                              xy_scale=self.stride, cmap='Oranges', fill=False,
                              regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    show.margins(ax, regression_fields[f, :6], xy_scale=self.stride)

                self.colorbar(ax, q)
