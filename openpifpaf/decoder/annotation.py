import numpy as np

# pylint: disable=import-error
from ..functional import scalar_value_clipped


class Annotation(object):
    def __init__(self, keypoints, skeleton, *, suppress_score_index=None):
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.suppress_score_index = suppress_score_index

        self.data = np.zeros((len(keypoints), 3), dtype=np.float32)
        self.joint_scales = np.zeros((len(keypoints),), dtype=np.float32)
        self.fixed_score = None
        self.decoding_order = []
        self.frontier_order = []

        self.skeleton_m1 = (np.asarray(skeleton) - 1).tolist()

        self.score_weights = np.ones((len(keypoints),))
        if self.suppress_score_index:
            self.score_weights[-1] = 0.0
        self.score_weights[:3] = 3.0
        self.score_weights /= np.sum(self.score_weights)

    def add(self, joint_i, xyv):
        self.data[joint_i] = xyv
        return self

    def rescale(self, scale_factor):
        self.data[:, 0:2] *= scale_factor
        if self.joint_scales is not None:
            self.joint_scales *= scale_factor
        for _, __, c1, c2 in self.decoding_order:
            c1[:2] *= scale_factor
            c2[:2] *= scale_factor
        return self

    def fill_joint_scales(self, scales, hr_scale=1.0):
        self.joint_scales = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale = scalar_value_clipped(scales[xyv_i], xyv[0] * hr_scale, xyv[1] * hr_scale)
            self.joint_scales[xyv_i] = scale / hr_scale

    def score(self):
        if self.fixed_score is not None:
            return self.fixed_score

        v = self.data[:, 2]
        if self.suppress_score_index is not None:
            v = np.copy(v)
            v[self.suppress_score_index] = 0.0
        # return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))
        # return np.sum(self.score_weights * np.sort(np.square(v))[::-1])
        return np.sum(self.score_weights * np.sort(v)[::-1])

    def scale(self, v_th=0.5):
        m = self.data[:, 2] > v_th
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )

    def json_data(self):
        """Data ready for json dump."""

        # convert to float64 before rounding because otherwise extra digits
        # will be added when converting to Python type
        return {
            'keypoints': np.around(self.data.astype(np.float64), 2).reshape(-1).tolist(),
            'bbox': [round(float(c), 2) for c in self.bbox()],
            'score': round(self.score(), 3),
        }

    def bbox(self):
        return self.bbox_from_keypoints(self.data, self.joint_scales)

    @staticmethod
    def bbox_from_keypoints(kps, joint_scales):
        m = kps[:, 2] > 0
        if not np.any(m):
            return [0, 0, 0, 0]

        x = np.min(kps[:, 0][m] - joint_scales[m])
        y = np.min(kps[:, 1][m] - joint_scales[m])
        w = np.max(kps[:, 0][m] + joint_scales[m]) - x
        h = np.max(kps[:, 1][m] + joint_scales[m]) - y
        return [x, y, w, h]


class AnnotationWithoutSkeleton(object):
    def __init__(self, j, xyv, n_joints):
        self.data = np.zeros((n_joints, 3))
        self.joint_scales = None
        self.data[j] = xyv

    def fill_joint_scales(self, scales, hr_scale):
        self.joint_scales = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale_field = scales[xyv_i]
            i = max(0, min(scale_field.shape[1] - 1, int(round(xyv[0] * hr_scale))))
            j = max(0, min(scale_field.shape[0] - 1, int(round(xyv[1] * hr_scale))))
            self.joint_scales[xyv_i] = scale_field[j, i] / hr_scale

    def score(self):
        v = self.data[:, 2]
        return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))

    def scale(self):
        m = self.data[:, 2] > 0.5
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )
