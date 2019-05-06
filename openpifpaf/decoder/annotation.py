import numpy as np


class Annotation(object):
    def __init__(self, j, xyv, skeleton):
        n_joints = len(set(i for c in skeleton for i in c))
        self.data = np.zeros((n_joints, 3))
        self.joint_scales = None
        self.data[j] = xyv
        self.fixed_score = None

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
        if self.fixed_score is not None:
            return self.fixed_score

        v = self.data[:, 2]
        return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))

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
