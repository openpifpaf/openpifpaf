import numpy as np


class AnnRescaler(object):
    def __init__(self, input_output_scale, n_keypoints):
        self.input_output_scale = input_output_scale
        self.n_keypoints = n_keypoints

    def __call__(self, anns, width_height_original):
        keypoint_sets = self.anns_to_keypoint_sets(anns)
        keypoint_sets[:, :, :2] /= self.input_output_scale

        # background mask
        bg_mask = self.anns_to_bg_mask(width_height_original, anns)
        bg_mask = bg_mask[::self.input_output_scale, ::self.input_output_scale]

        # valid area
        valid_area = None
        if anns and 'valid_area' in anns[0]:
            valid_area = anns[0]['valid_area']
            valid_area = (
                valid_area[0] / self.input_output_scale,
                valid_area[1] / self.input_output_scale,
                valid_area[2] / self.input_output_scale,
                valid_area[3] / self.input_output_scale,
            )

        return keypoint_sets, bg_mask, valid_area

    def anns_to_keypoint_sets(self, anns):
        """Ignore annotations of crowds."""
        keypoint_sets = [ann['keypoints'] for ann in anns if not ann['iscrowd']]
        if not keypoint_sets:
            return np.zeros((0, self.n_keypoints, 3))

        return np.stack(keypoint_sets)

    @staticmethod
    def anns_to_bg_mask(width_height, anns, include_annotated=True):
        """Create background mask taking crowded annotations into account."""
        mask = np.ones(width_height[::-1], dtype=np.bool)
        for ann in anns:
            if include_annotated and \
            not ann['iscrowd'] and \
            'keypoints' in ann and \
            np.any(ann['keypoints'][:, 2] > 0):
                continue

            if 'mask' not in ann:
                bb = ann['bbox'].copy()
                bb[2:] += bb[:2]  # convert width and height to x2 and y2
                bb[0] = np.clip(bb[0], 0, mask.shape[1] - 1)
                bb[1] = np.clip(bb[1], 0, mask.shape[0] - 1)
                bb[2] = np.clip(bb[2], 0, mask.shape[1] - 1)
                bb[3] = np.clip(bb[3], 0, mask.shape[0] - 1)
                bb = bb.astype(np.int)
                mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1] = 0
                continue
            mask[ann['mask']] = 0
        return mask
