import numpy as np

# pylint: disable=import-error
from ..functional import scalar_value_clipped


class Annotation(object):
    def __init__(self, keypoints, skeleton):
        self.keypoints = keypoints
        self.skeleton = skeleton

        self.data = np.zeros((len(keypoints), 3), dtype=np.float32)
        self.joint_scales = None
        self.fixed_score = None
        self.decoding_order = []

        self.skeleton_m1 = (np.asarray(skeleton) - 1).tolist()

        self.score_weights = np.ones((len(keypoints),))
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
        # return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))
        # return np.sum(self.score_weights * np.sort(np.square(v))[::-1])
        return np.sum(self.score_weights * np.sort(v)[::-1])

    def frontier(self):
        """Frontier to complete annotation.

        Format: (
            confidence of origin,
            connection index,
            forward?,
            joint index 1,  (not corrected for forward)
            joint index 2,  (not corrected for forward)
        )
        """
        return sorted([
            (self.data[j1i, 2], connection_i, True, j1i, j2i)
            for connection_i, (j1i, j2i) in enumerate(self.skeleton_m1)
            if self.data[j1i, 2] > 0.0 and self.data[j2i, 2] == 0.0
        ] + [
            (self.data[j2i, 2], connection_i, False, j1i, j2i)
            for connection_i, (j1i, j2i) in enumerate(self.skeleton_m1)
            if self.data[j2i, 2] > 0.0 and self.data[j1i, 2] == 0.0
        ], reverse=True)

    def frontier_iter(self):
        frontier = list(self.frontier())
        while frontier:
            next_item = frontier.pop(0)
            forward = next_item[2]
            i_target = next_item[4] if forward else next_item[3]
            xyv_target = self.data[i_target]

            if xyv_target[2] != 0.0:
                # another frontier connection has filled this joint
                continue

            yield next_item

            if xyv_target[2] == 0.0:
                # No connection created. Done.
                continue

            # Need to add connections starting from the new joint to the frontier.
            frontier += [
                (self.data[j1i, 2], connection_i, True, j1i, j2i)
                for connection_i, (j1i, j2i) in enumerate(self.skeleton_m1)
                if j1i == i_target and self.data[j2i, 2] == 0.0
            ] + [
                (self.data[j2i, 2], connection_i, False, j1i, j2i)
                for connection_i, (j1i, j2i) in enumerate(self.skeleton_m1)
                if j2i == i_target and self.data[j1i, 2] == 0.0
            ]
            frontier = list(sorted(frontier, reverse=True))

    def scale(self, v_th=0.5):
        m = self.data[:, 2] > v_th
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )


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
