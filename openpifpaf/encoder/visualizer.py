import numpy as np

from ..data import COCO_KEYPOINTS, COCO_PERSON_SKELETON
from .. import show


class Visualizer(object):
    def __init__(self, headnames, strides):
        self.headnames = headnames
        self.strides = strides
        self.keypoint_painter = show.KeypointPainter(skeleton=COCO_PERSON_SKELETON)

    def single(self, image, targets):
        keypoint_sets = None
        if 'skeleton' in self.headnames:
            i = self.headnames.index('skeleton')
            keypoint_sets = targets[i][0]

        with show.canvas() as ax:
            ax.imshow(image)

        for target, headname, stride in zip(targets, self.headnames, self.strides):
            print(headname, len(target))
            if headname in ('paf', 'paf19', 'pafs', 'wpaf'):
                self.paf19(image, target, stride, keypoint_sets)
            elif headname in ('pif', 'pif17', 'pifs'):
                self.pif17(image, target, stride, keypoint_sets)

    def pif17(self, image, target, stride, keypoint_sets):
        resized_image = image[::stride, ::stride]
        for f in [1, 2, 15, 16]:
            print('intensity field', COCO_KEYPOINTS[f])

            with show.canvas() as ax:
                ax.imshow(resized_image)
                bce_mask = np.sum(target[0], axis=0) > 0.0
                ax.imshow(target[0][f] + 0.5 * bce_mask, alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets)
                show.quiver(ax, target[1][f], xy_scale=stride)

    def paf19(self, image, target, stride, keypoint_sets):
        resized_image = image[::stride, ::stride]
        for f in [1, 2, 15, 16, 17, 18]:
            print('association field',
                  COCO_KEYPOINTS[COCO_PERSON_SKELETON[f][0] - 1],
                  COCO_KEYPOINTS[COCO_PERSON_SKELETON[f][1] - 1])
            with show.canvas() as ax:
                ax.imshow(resized_image)
                bce_mask = np.sum(target[0], axis=0) > 0.0
                ax.imshow(target[0][f] + 0.5 * bce_mask, alpha=0.9, vmin=0.0, vmax=1.0)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets)
                show.quiver(ax, target[1][f], xy_scale=stride)

            with show.canvas() as ax:
                ax.imshow(image)
                show.white_screen(ax, alpha=0.5)
                self.keypoint_painter.keypoints(ax, keypoint_sets)
                show.quiver(ax, target[2][f], xy_scale=stride)

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
