import numpy as np


class Annotation(object):
    def __init__(self, j, xyv, skeleton):
        n_joints = len(set(i for c in skeleton for i in c))
        self.data = np.zeros((n_joints, 3))
        self.joint_scales = None
        self.data[j] = xyv

        self.skeleton_m1 = (np.asarray(skeleton) - 1).tolist()

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
        # return 0.5 * np.max(v) + 0.5 * np.mean(v)
        return np.mean(np.square(v))

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
        block_frontier = set()
        while True:
            unblocked_frontier = [f for f in self.frontier()
                                  if (f[1], f[2]) not in block_frontier]
            if not unblocked_frontier:
                break

            first = unblocked_frontier[0]
            yield first
            block_frontier.add((first[1], first[2]))

    def scale(self):
        m = self.data[:, 2] > 0.5
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
        # return 0.5 * np.max(v) + 0.5 * np.mean(v)
        return np.mean(np.square(v))

    def scale(self):
        m = self.data[:, 2] > 0.5
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )
