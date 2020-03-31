import logging

import numpy as np

LOG = logging.getLogger(__name__)


class AnnRescaler(object):
    def __init__(self, stride, n_keypoints, davinci_pose):
        self.stride = stride
        self.n_keypoints = n_keypoints
        self.davinci_pose = davinci_pose
        self.davinci_total_area = (
            (np.max(self.davinci_pose[:, 0]) - np.min(self.davinci_pose[:, 0])) *
            (np.max(self.davinci_pose[:, 1]) - np.min(self.davinci_pose[:, 1]))
        )

        # rotate the davinci pose by 45 degrees
        c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
        rotate = np.array(((c, -s), (s, c)))
        self.davinci_pose_45 = np.copy(davinci_pose)
        self.davinci_pose_45[:, :2] = np.einsum('ij,kj->ki', rotate, self.davinci_pose_45[:, :2])
        self.davinci_45_total_area = (
            (np.max(self.davinci_pose_45[:, 0]) - np.min(self.davinci_pose_45[:, 0])) *
            (np.max(self.davinci_pose_45[:, 1]) - np.min(self.davinci_pose_45[:, 1]))
        )

    def valid_area(self, meta):
        if 'valid_area' not in meta:
            return None

        return (
            meta['valid_area'][0] / self.stride,
            meta['valid_area'][1] / self.stride,
            meta['valid_area'][2] / self.stride,
            meta['valid_area'][3] / self.stride,
        )

    def keypoint_sets(self, anns):
        """Ignore annotations of crowds."""
        keypoint_sets = [ann['keypoints'] for ann in anns if not ann['iscrowd']]
        if not keypoint_sets:
            return np.zeros((0, self.n_keypoints, 3))

        keypoint_sets = np.stack(keypoint_sets)
        keypoint_sets[:, :, :2] /= self.stride
        return keypoint_sets

    def bg_mask(self, anns, width_height, *, include_annotated=True):
        """Create background mask taking crowded annotations into account."""
        mask = np.ones((
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)
        for ann in anns:
            if include_annotated and \
               not ann['iscrowd'] and \
               'keypoints' in ann and \
               np.any(ann['keypoints'][:, 2] > 0):
                continue

            if 'mask' not in ann:
                bb = ann['bbox'].copy()
                bb /= self.stride
                bb[2:] += bb[:2]  # convert width and height to x2 and y2
                left = np.clip(int(bb[0]), 0, mask.shape[1] - 1)
                top = np.clip(int(bb[1]), 0, mask.shape[0] - 1)
                right = np.clip(int(np.ceil(bb[2])), left + 1, mask.shape[1])
                bottom = np.clip(int(np.ceil(bb[3])), top + 1, mask.shape[0])
                mask[top:bottom, left:right] = 0
                continue

            assert False  # because code below is not tested
            mask[ann['mask'][::self.stride, ::self.stride]] = 0

        return mask

    def scale(self, keypoints):
        visible = keypoints[:, 2] > 0
        if np.sum(visible) < 3:
            return np.nan

        area = (
            (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
            (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        area_davinci = (
            (np.max(self.davinci_pose[visible, 0]) - np.min(self.davinci_pose[visible, 0])) *
            (np.max(self.davinci_pose[visible, 1]) - np.min(self.davinci_pose[visible, 1]))
        )
        area_davinci_45 = (
            (np.max(self.davinci_pose_45[visible, 0]) - np.min(self.davinci_pose_45[visible, 0])) *
            (np.max(self.davinci_pose_45[visible, 1]) - np.min(self.davinci_pose_45[visible, 1]))
        )
        factor = np.sqrt(min(self.davinci_total_area / area_davinci,
                             self.davinci_45_total_area / area_davinci_45))
        factor_clipped = min(5.0, factor)
        scale = np.sqrt(area) * factor_clipped
        if scale < 0.1:
            scale = np.nan

        LOG.debug('instance scale = %.3f (factor = %.2f, clipped factor = %.2f)',
                  scale, factor, factor_clipped)
        return scale
