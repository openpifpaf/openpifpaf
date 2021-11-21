import logging

import numpy as np

LOG = logging.getLogger(__name__)


class AnnRescaler():
    suppress_selfhidden = True
    suppress_invisible = False
    suppress_collision = False

    def __init__(self, stride, pose=None):
        self.stride = stride
        self.pose = pose

        self.pose_total_area = None
        self.pose_45 = None
        self.pose_45_total_area = None
        if pose is not None:
            self.pose_total_area = (
                (np.max(self.pose[:, 0]) - np.min(self.pose[:, 0]))
                * (np.max(self.pose[:, 1]) - np.min(self.pose[:, 1]))
            )

            # rotate the davinci pose by 45 degrees
            c, s = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
            rotate = np.array(((c, -s), (s, c)))
            self.pose_45 = np.copy(self.pose)
            self.pose_45[:, :2] = np.einsum('ij,kj->ki', rotate, self.pose_45[:, :2])
            self.pose_45_total_area = (
                (np.max(self.pose_45[:, 0]) - np.min(self.pose_45[:, 0]))
                * (np.max(self.pose_45[:, 1]) - np.min(self.pose_45[:, 1]))
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

    @staticmethod
    def suppress_collision_(keypoint_sets_bbox):
        for p_i, (kps_p, bbox_p) in enumerate(keypoint_sets_bbox[:-1]):
            for kps_s, bbox_s in keypoint_sets_bbox[p_i + 1:]:
                d_th = 0.2 * max(bbox_p[2], bbox_p[3], bbox_s[2], bbox_s[3])
                d_th = max(16.0, d_th)
                diff = np.abs(kps_p[:, :2] - kps_s[:, :2])
                collision = (
                    (kps_p[:, 2] > 0.0)
                    & (kps_s[:, 2] > 0.0)
                    & (diff[:, 0] < d_th)
                    & (diff[:, 1] < d_th)
                )
                if np.any(collision):
                    kps_p[collision, 2] = 0.0
                    kps_s[collision, 2] = 0.0

    @staticmethod
    def suppress_selfhidden_(keypoint_sets):
        for kpi in range(len(keypoint_sets[0])):
            all_xyv = sorted([keypoints[kpi] for keypoints in keypoint_sets],
                             key=lambda xyv: xyv[2], reverse=True)
            for i, xyv in enumerate(all_xyv[1:], start=1):
                if xyv[2] > 1.0:  # is visible
                    continue
                if xyv[2] < 1.0:  # does not exist
                    break
                for prev_xyv in all_xyv[:i]:
                    if prev_xyv[2] <= 1.0:  # do not suppress if both hidden
                        break
                    if np.abs(prev_xyv[0] - xyv[0]) > 32.0 \
                       or np.abs(prev_xyv[1] - xyv[1]) > 32.0:
                        continue
                    LOG.debug('suppressing %s for %s (kp %d)', xyv, prev_xyv, i)
                    xyv[2] = 0.0
                    break  # only need to suppress a keypoint once

    def keypoint_sets(self, anns):
        """Ignore annotations of crowds."""
        keypoint_sets_bbox = [(np.copy(ann['keypoints']), ann['bbox'])
                              for ann in anns if not ann['iscrowd']]
        if not keypoint_sets_bbox:
            return []

        if self.suppress_collision:
            self.suppress_collision_(keypoint_sets_bbox)
        keypoint_sets = [kps for kps, _ in keypoint_sets_bbox]

        if self.suppress_invisible:
            for kps in keypoint_sets:
                kps[kps[:, 2] < 2.0, 2] = 0.0
        elif self.suppress_selfhidden:
            self.suppress_selfhidden_(keypoint_sets)

        for keypoints in keypoint_sets:
            keypoints[:, :2] /= self.stride
        return keypoint_sets

    def bg_mask(self, anns, width_height, *, crowd_margin):
        """Create background mask taking crowd annotations into account."""
        mask = np.ones((
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)
        for ann in anns:
            if not ann['iscrowd']:
                valid_keypoints = 'keypoints' in ann and np.any(ann['keypoints'][:, 2] > 0)
                if valid_keypoints:
                    continue

            if 'mask' not in ann:
                bb = ann['bbox'].copy()
                bb /= self.stride
                bb[2:] += bb[:2]  # convert width and height to x2 and y2

                # left top
                left = np.clip(int(bb[0] - crowd_margin), 0, mask.shape[1] - 1)
                top = np.clip(int(bb[1] - crowd_margin), 0, mask.shape[0] - 1)

                # right bottom
                # ceil: to round up
                # +1: because mask upper limit is exclusive
                right = np.clip(int(np.ceil(bb[2] + crowd_margin)) + 1,
                                left + 1, mask.shape[1])
                bottom = np.clip(int(np.ceil(bb[3] + crowd_margin)) + 1,
                                 top + 1, mask.shape[0])

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
            (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0]))
            * (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        factor = 1.0

        if self.pose is not None:
            area_ref = (
                (np.max(self.pose[visible, 0]) - np.min(self.pose[visible, 0]))
                * (np.max(self.pose[visible, 1]) - np.min(self.pose[visible, 1]))
            )
            area_ref_45 = (
                (np.max(self.pose_45[visible, 0]) - np.min(self.pose_45[visible, 0]))
                * (np.max(self.pose_45[visible, 1]) - np.min(self.pose_45[visible, 1]))
            )

            factor = np.sqrt(min(
                self.pose_total_area / area_ref if area_ref > 0.1 else np.inf,
                self.pose_45_total_area / area_ref_45 if area_ref_45 > 0.1 else np.inf,
            ))
            if np.isinf(factor):
                return np.nan

        factor_clipped = min(5.0, factor)
        scale = np.sqrt(area) * factor_clipped
        if scale < 0.1:
            scale = np.nan

        LOG.debug('instance scale = %.3f (factor = %.2f, clipped factor = %.2f)',
                  scale, factor, factor_clipped)
        return scale


class AnnRescalerDet():
    def __init__(self, stride, n_categories):
        self.stride = stride
        self.n_categories = n_categories

    def valid_area(self, meta):
        if 'valid_area' not in meta:
            return None

        return (
            meta['valid_area'][0] / self.stride,
            meta['valid_area'][1] / self.stride,
            meta['valid_area'][2] / self.stride,
            meta['valid_area'][3] / self.stride,
        )

    def detections(self, anns):
        category_bboxes = [(ann['category_id'], ann['bbox'] / self.stride)
                           for ann in anns if not ann['iscrowd']]
        return category_bboxes

    def bg_mask(self, anns, width_height, *, crowd_margin):
        """Create background mask taking crowd annotations into account."""
        mask = np.ones((
            self.n_categories,
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)
        for ann in anns:
            if not ann['iscrowd']:
                continue

            if 'mask' not in ann:
                field_i = ann['category_id'] - 1
                bb = ann['bbox'].copy()
                bb /= self.stride
                bb[2:] += bb[:2]  # convert width and height to x2 and y2
                left = np.clip(int(bb[0] - crowd_margin), 0, mask.shape[1] - 1)
                top = np.clip(int(bb[1] - crowd_margin), 0, mask.shape[0] - 1)
                right = np.clip(int(np.ceil(bb[2] + crowd_margin)) + 1,
                                left + 1, mask.shape[1])
                bottom = np.clip(int(np.ceil(bb[3] + crowd_margin)) + 1,
                                 top + 1, mask.shape[0])
                mask[field_i, top:bottom, left:right] = 0
                continue

            assert False  # because code below is not tested
            mask[ann['mask'][::self.stride, ::self.stride]] = 0

        return mask


class TrackingAnnRescaler(AnnRescaler):
    def bg_mask(self, anns, width_height, *, crowd_margin):
        """Create background mask taking crowd annotations into account."""
        anns1, anns2 = anns

        mask = np.ones((
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)
        crowd_bbox = [np.inf, np.inf, 0, 0]
        for ann in anns1 + anns2:
            if not ann['iscrowd']:
                valid_keypoints = 'keypoints' in ann and np.any(ann['keypoints'][:, 2] > 0)
                if valid_keypoints:
                    continue

            if 'mask' not in ann:
                bb = ann['bbox'].copy()
                bb /= self.stride
                bb[2:] += bb[:2]  # convert width and height to x2 and y2

                # left top
                left = np.clip(int(bb[0] - crowd_margin), 0, mask.shape[1] - 1)
                top = np.clip(int(bb[1] - crowd_margin), 0, mask.shape[0] - 1)

                # right bottom
                # ceil: to round up
                # +1: because mask upper limit is exclusive
                right = np.clip(int(np.ceil(bb[2] + crowd_margin)) + 1,
                                left + 1, mask.shape[1])
                bottom = np.clip(int(np.ceil(bb[3] + crowd_margin)) + 1,
                                 top + 1, mask.shape[0])

                crowd_bbox[0] = min(crowd_bbox[0], left)
                crowd_bbox[1] = min(crowd_bbox[1], top)
                crowd_bbox[2] = max(crowd_bbox[2], right)
                crowd_bbox[3] = max(crowd_bbox[3], bottom)
                continue

            assert False  # because code below is not tested
            mask[ann['mask'][::self.stride, ::self.stride]] = 0

        if crowd_bbox[1] < crowd_bbox[3] and crowd_bbox[0] < crowd_bbox[2]:
            LOG.debug('crowd_bbox: %s', crowd_bbox)
            mask[crowd_bbox[1]:crowd_bbox[3], crowd_bbox[0]:crowd_bbox[2]] = 0

        return mask

    def keypoint_sets(self, anns):
        """Ignore annotations of crowds."""
        anns1, anns2 = anns

        anns1_by_trackid = {ann['track_id']: ann for ann in anns1}
        keypoint_sets_bbox = [
            (
                np.concatenate((
                    anns1_by_trackid[ann2['track_id']]['keypoints'],
                    ann2['keypoints'],
                ), axis=0),
                ann2['bbox'],
            )
            for ann2 in anns2
            if (not ann2['iscrowd']
                and ann2['track_id'] in anns1_by_trackid)
        ]
        if not keypoint_sets_bbox:
            return []

        if self.suppress_collision:
            self.suppress_collision_(keypoint_sets_bbox)
        keypoint_sets = [kps for kps, _ in keypoint_sets_bbox]

        if self.suppress_invisible:
            for kps in keypoint_sets:
                kps[kps[:, 2] < 2.0, 2] = 0.0

        for keypoints in keypoint_sets:
            keypoints[:, :2] /= self.stride
        return keypoint_sets
