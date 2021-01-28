import numpy as np

# pylint: disable=import-error
from .functional import scalar_value_clipped

NOTSET = '__notset__'


class Annotation:
    def __init__(self, keypoints, skeleton, *, category_id=1, suppress_score_index=None):
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.category_id = category_id
        self.suppress_score_index = suppress_score_index

        self.data = np.zeros((len(keypoints), 3), dtype=np.float32)
        self.joint_scales = np.zeros((len(keypoints),), dtype=np.float32)
        self.fixed_score = NOTSET
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

    def set(self, data, joint_scales=None, *, fixed_score=NOTSET):
        self.data = data
        if joint_scales is not None:
            self.joint_scales = joint_scales
        else:
            self.joint_scales[:] = 0.0
        self.fixed_score = fixed_score
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
        if self.fixed_score != NOTSET:
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

        # avoid visible keypoints becoming invisible due to rounding
        v_mask = self.data[:, 2] > 0.0
        keypoints = np.copy(self.data)
        keypoints[v_mask, 2] = np.maximum(0.01, keypoints[v_mask, 2])
        keypoints = np.around(keypoints.astype(np.float64), 2)

        # convert to float64 before rounding because otherwise extra digits
        # will be added when converting to Python type
        data = {
            'keypoints': keypoints.reshape(-1).tolist(),
            'bbox': [round(float(c), 2) for c in self.bbox()],
            'score': max(0.001, round(self.score(), 3)),
            'category_id': self.category_id,
        }

        id_ = getattr(self, 'id_', None)
        if id_:
            data['id_'] = id_

        return data

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


class AnnotationDet:
    def __init__(self, categories):
        self.categories = categories
        self.field_i = None
        self.score = None
        self.bbox = None

    def set(self, field_i, score, bbox):
        """Set score to None for a ground truth annotation."""
        self.field_i = field_i
        self.score = score
        self.bbox = np.asarray(bbox)
        return self

    @property
    def category(self):
        return self.categories[self.field_i]

    def json_data(self):
        return {
            'category_id': self.field_i + 1,
            'category': self.category,
            'score': max(0.001, round(float(self.score), 3)),
            'bbox': [round(float(c), 2) for c in self.bbox],
        }
