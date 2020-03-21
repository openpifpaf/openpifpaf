import logging
import numpy as np

from .. import show

LOG = logging.getLogger(__name__)


class CifVisualizer(object):
    show_margin = False

    def __init__(self, head_name, stride, indices, *, keypoints, skeleton):
        self.head_name = head_name
        self.stride = stride
        self.indices = indices

        self.keypoints = keypoints
        self.skeleton = skeleton

        self.keypoint_painter = show.KeypointPainter(xy_scale=self.stride)
        LOG.debug('%s: indices = %s', head_name, self.indices)

    def __call__(self, image, target, meta, *, keypoint_sets=None):
        image = np.moveaxis(np.asarray(image), 0, -1)
        image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)

        resized_image = image[::self.stride, ::self.stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        for f in self.indices:
            LOG.debug('intensity field %s', self.keypoints[f])

            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=self.skeleton)
                show.quiver(ax, target[1][f, :2], xy_scale=self.stride, uv_is_offset=True)
                show.boxes(ax, target[2][f], regression_field=target[1][f, :2],
                           xy_scale=self.stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=True)
                if self.show_margin:
                    show.margins(ax, target[1][f, :6], xy_scale=self.stride)


class CafVisualizer(object):
    show_margin = False

    def __init__(self, head_name, stride, indices, *, keypoints, skeleton):
        self.head_name = head_name
        self.stride = stride
        self.indices = indices

        self.keypoints = keypoints
        self.skeleton = skeleton

        self.keypoint_painter = show.KeypointPainter(xy_scale=self.stride)
        LOG.debug('%s: indices = %s', head_name, self.indices)

    def __call__(self, image, target, meta, *, keypoint_sets=None):
        image = np.moveaxis(np.asarray(image), 0, -1)
        image = np.clip((image + 2.0) / 4.0, 0.0, 1.0)

        resized_image = image[::self.stride, ::self.stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        for f in self.indices:
            LOG.debug('association field %s,%s',
                      self.keypoints[self.skeleton[f][0] - 1],
                      self.keypoints[self.skeleton[f][1] - 1])
            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=self.skeleton)
                show.quiver(ax, target[1][f, :2], xy_scale=self.stride, uv_is_offset=True)
                show.boxes(ax, target[3][f], regression_field=target[1][f, :2],
                           xy_scale=self.stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=True)
                if self.show_margin:
                    show.margins(ax, target[1][f, :6], xy_scale=self.stride)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=self.skeleton)
                show.quiver(ax, target[2][f, :2], xy_scale=self.stride, uv_is_offset=True)
                show.boxes(ax, target[4][f], regression_field=target[2][f, :2],
                           xy_scale=self.stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=True)
                if self.show_margin:
                    show.margins(ax, target[2][f, :6], xy_scale=self.stride)
