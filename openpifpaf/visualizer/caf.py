import copy
import logging

from .base import Base
from ..annotation import Annotation
from .. import headmeta, show

try:
    import matplotlib.cm
    CMAP_BLUES_NAN = copy.copy(matplotlib.cm.get_cmap('Blues'))
    CMAP_BLUES_NAN.set_bad('white', alpha=0.5)
except ImportError:
    CMAP_BLUES_NAN = None

LOG = logging.getLogger(__name__)


class Caf(Base):
    def __init__(self, meta: headmeta.Caf):
        super().__init__(meta.name)
        self.meta = meta
        keypoint_painter = show.KeypointPainter(monocolor_connections=True)
        self.annotation_painter = show.AnnotationPainter(painters={'Annotation': keypoint_painter})

    def targets(self, field, *, annotation_dicts):
        assert self.meta.keypoints is not None
        assert self.meta.skeleton is not None

        annotations = [
            Annotation(
                keypoints=self.meta.keypoints,
                skeleton=self.meta.skeleton,
                sigmas=self.meta.sigmas,
            ).set(
                ann['keypoints'], fixed_score='', fixed_bbox=ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5], field[:, 7], field[:, 8],
                          annotations=annotations)

    def predicted(self, field):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5], field[:, 7], field[:, 8],
                          annotations=self._ground_truth,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        for f in self.indices('confidence'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        for f in self.indices('regression'):
            LOG.debug('%s,%s',
                      self.meta.keypoints[self.meta.skeleton[f][0] - 1],
                      self.meta.keypoints[self.meta.skeleton[f][1] - 1])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.annotation_painter.annotations(ax, annotations, color='lightgray')
                q1 = show.quiver(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_field,
                                 xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                 cmap='Blues', clim=(0.5, 1.0), width=0.001)
                show.quiver(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_field,
                            xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001)
                show.boxes(ax, scale_fields1[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields1[f, :2],
                           xy_scale=self.meta.stride, cmap='Blues', fill=False,
                           regression_field_is_offset=uv_is_offset)
                show.boxes(ax, scale_fields2[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields2[f, :2],
                           xy_scale=self.meta.stride, cmap='Greens', fill=False,
                           regression_field_is_offset=uv_is_offset)

                self.colorbar(ax, q1)
