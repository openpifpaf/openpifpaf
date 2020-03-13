import logging
import numpy as np

from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_CONNECTIONS
from .. import show

LOG = logging.getLogger(__name__)


class Visualizer(object):
    pif_indices = []
    paf_indices = []
    dpaf_indices = []

    def __init__(self, head_names, strides, *,
                 show_margin=False):
        self.head_names = head_names
        self.strides = strides
        self.show_margin = show_margin

        self.keypoint_painter = show.KeypointPainter()
        LOG.debug('pif = %s, paf = %s, dpaf = %s',
                  self.pif_indices, self.paf_indices, self.dpaf_indices)

    def single(self, image, targets):
        assert len(targets) == len(self.head_names) + 1  # skeleton is last
        keypoint_sets = targets[-1][0]

        with show.canvas() as ax:
            ax.imshow(image)

        for target, headname, stride in zip(targets, self.head_names, self.strides):
            LOG.debug('%s with %d components', headname, len(target))
            if headname in ('paf', 'paf19', 'pafs', 'wpaf'):
                self.paf(image, target, stride, keypoint_sets,
                         indices=self.paf_indices,
                         keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
            elif headname in ('pif', 'pif17', 'pifs'):
                self.pif(image, target, stride, keypoint_sets,
                         indices=self.pif_indices,
                         keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
            elif headname in ('paf25', 'pafs25'):
                self.paf(image, target, stride, keypoint_sets,
                         indices=self.dpaf_indices,
                         keypoints=COCO_KEYPOINTS, skeleton=DENSER_COCO_PERSON_CONNECTIONS)
            else:
                LOG.warning('unknown head: %s', headname)

    def pif(self, image, target, stride, keypoint_sets, *, indices, keypoints, skeleton):
        resized_image = image[::stride, ::stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        for f in indices:
            LOG.debug('intensity field %s', keypoints[f])

            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=skeleton)
                show.quiver(ax, target[1][f, :2], xy_scale=stride, uv_is_offset=True)
                show.boxes(ax, target[2][f], regression_field=target[1][f, :2],
                           xy_scale=stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=True)
                if self.show_margin:
                    show.margins(ax, target[1][f, :6], xy_scale=stride)

    def paf(self, image, target, stride, keypoint_sets, *, indices, keypoints, skeleton):
        resized_image = image[::stride, ::stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        for f in indices:
            LOG.debug('association field %s,%s',
                      keypoints[skeleton[f][0] - 1],
                      keypoints[skeleton[f][1] - 1])
            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=skeleton)
                show.quiver(ax, target[1][f, :2], xy_scale=stride, uv_is_offset=True)
                show.boxes(ax, target[3][f], regression_field=target[1][f, :2],
                           xy_scale=stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=True)
                if self.show_margin:
                    show.margins(ax, target[1][f, :6], xy_scale=stride)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=skeleton)
                show.quiver(ax, target[2][f, :2], xy_scale=stride, uv_is_offset=True)
                show.boxes(ax, target[4][f], regression_field=target[2][f, :2],
                           xy_scale=stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=True)
                if self.show_margin:
                    show.margins(ax, target[2][f, :6], xy_scale=stride)

    def __call__(self, images, targets, meta):
        n_heads = len(targets)
        n_batch = len(images)
        targets = [[t.numpy() for t in heads] for heads in targets]
        targets = [
            [[target_field[batch_i] for target_field in targets[head_i]]
             for head_i in range(n_heads)]
            for batch_i in range(n_batch)
        ]

        images = np.moveaxis(np.asarray(images), 1, -1)
        images = np.clip((images + 2.0) / 4.0, 0.0, 1.0)

        for i, t in zip(images, targets):
            self.single(i, t)
