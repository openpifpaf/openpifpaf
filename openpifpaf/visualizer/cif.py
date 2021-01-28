import logging

from .base import BaseVisualizer
from ..annotation import Annotation
from .. import show

LOG = logging.getLogger(__name__)


class Cif(BaseVisualizer):
    show_margin = False
    show_confidences = False
    show_regressions = False
    show_background = False

    def __init__(self, head_name, *, stride=1, keypoints=None, skeleton=None):
        super().__init__(head_name)

        self.stride = stride
        self.keypoints = keypoints
        self.skeleton = skeleton

        self.keypoint_painter = show.KeypointPainter(xy_scale=self.stride)

    def targets(self, field, keypoint_sets):
        assert self.keypoints is not None
        assert self.skeleton is not None

        annotations = [
            Annotation(keypoints=self.keypoints, skeleton=self.skeleton).set(kps, fixed_score=None)
            for kps in keypoint_sets
        ]

        self._confidences(field[0])
        self._regressions(field[1], field[2], annotations=annotations)

    def predicted(self, field, *, annotations=None):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 4],
                          annotations=annotations,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return

        for f in self.indices:
            LOG.debug('%s', self.keypoints[f])

            with self.image_canvas(self._processed_image) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, scale_fields, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return

        for f in self.indices:
            LOG.debug('%s', self.keypoints[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.keypoint_painter.annotations(ax, annotations)
                q = show.quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.stride, uv_is_offset=uv_is_offset,
                                cmap='Oranges', clim=(0.5, 1.0), width=0.001)
                show.boxes(ax, scale_fields[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields[f, :2],
                           xy_scale=self.stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    show.margins(ax, regression_fields[f, :6], xy_scale=self.stride)

                self.colorbar(ax, q)
