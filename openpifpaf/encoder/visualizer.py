import logging
import numpy as np

from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON, DENSER_COCO_PERSON_CONNECTIONS
from .. import show

LOG = logging.getLogger(__name__)


class Visualizer(object):
    def __init__(self, head_names, strides, *,
                 pif_indices=None, paf_indices=None,
                 show_margin=False):
        self.head_names = head_names
        self.strides = strides
        self.pif_indices = pif_indices or []
        self.paf_indices = paf_indices or []
        self.show_margin = show_margin

        self.keypoint_painter = show.KeypointPainter()

    def single(self, image, targets):
        keypoint_sets = None
        if 'skeleton' in self.head_names:
            i = self.head_names.index('skeleton')
            keypoint_sets = targets[i][0]

        with show.canvas() as ax:
            ax.imshow(image)

        for target, headname, stride in zip(targets, self.head_names, self.strides):
            LOG.debug('%s with %d components', headname, len(target))
            if headname in ('paf', 'paf19', 'pafs', 'wpaf'):
                self.paf(image, target, stride, keypoint_sets,
                         keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
            elif headname in ('pif', 'pif17', 'pifs'):
                self.pif(image, target, stride, keypoint_sets,
                         keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
            elif headname in ('paf25',):
                self.paf(image, target, stride, keypoint_sets,
                         keypoints=COCO_KEYPOINTS, skeleton=DENSER_COCO_PERSON_CONNECTIONS)
            else:
                LOG.warning('unknown head: %s', headname)

    def pif(self, image, target, stride, keypoint_sets, *, keypoints, skeleton):
        resized_image = image[::stride, ::stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        for f in self.pif_indices:
            LOG.debug('intensity field %s', keypoints[f])

            with show.canvas() as ax:
                ax.imshow(resized_image)
                ax.imshow(target[0][f] + 0.5 * bce_masks[f], alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=skeleton)
                show.quiver(ax, target[1][f, :2], xy_scale=stride)
                if self.show_margin:
                    show.margins(ax, target[1][f, :6], xy_scale=stride)

    def paf(self, image, target, stride, keypoint_sets, *, keypoints, skeleton):
        resized_image = image[::stride, ::stride]
        bce_targets = target[0]
        bce_masks = (bce_targets[:-1] + bce_targets[-1:]) > 0.5
        for f in self.paf_indices:
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
                show.quiver(ax, target[1][f, :2], xy_scale=stride)
                if self.show_margin:
                    show.margins(ax, target[1][f, :6], xy_scale=stride)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets, skeleton=skeleton)
                show.quiver(ax, target[2][f, :2], xy_scale=stride)
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
