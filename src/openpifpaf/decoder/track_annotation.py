import numpy as np


class TrackAnnotation:
    track_id_counter = 0

    def __init__(self):
        self.frame_pose = []

        TrackAnnotation.track_id_counter += 1
        self.id_ = TrackAnnotation.track_id_counter

    def add(self, frame_number, pose_annotation):
        self.frame_pose.append((frame_number, pose_annotation))
        return self

    def pose(self, frame_number):
        for frame_i, pose in reversed(self.frame_pose):
            if frame_i < frame_number:
                break
            if frame_i > frame_number:
                continue

            return pose

        return None

    def pose_score(self, frame_number):
        pose = self.pose(frame_number)
        if pose is None:
            return 0.0

        kps_order = np.argsort(pose.data[:, 2])[::-1]
        if pose.data[kps_order[1], 2] < 0.05:
            return 0.0

        pose.score_weights[:] = 1.0
        pose.score_weights[1] = 3.0
        pose.score_weights[2] = 5.0
        pose.score_weights[5:] = 0.1
        pose.score_weights[-2:] = 0.0  # ears are not annotated
        pose.score_weights /= np.sum(pose.score_weights)

        return pose.score

    def score(self, frame_number, current_importance=1.0):
        """Reduce current importance to rank tracks that are still processing
        for the current frame."""
        weights = [1.0 for _ in range(12)]
        weights[0] = current_importance
        return (
            sum(w * self.pose_score(frame_number - i) for i, w in enumerate(weights))
            / sum(weights)
        )

    def __len__(self):
        return len(self.frame_pose)
