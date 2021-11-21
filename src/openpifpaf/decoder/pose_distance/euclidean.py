import numpy as np


class Euclidean:
    """Compute Euclidean distance between a track and a new pose candidate."""

    invisible_penalty = 110.0

    def __init__(self, *, track_frames=None):
        if track_frames is None:
            track_frames = [-1]
        assert all(t < 0 for t in track_frames)

        self.valid_keypoints = None
        self.track_frames = track_frames

    def __call__(self, frame_number, pose, track, track_is_good):
        return min(
            self.distance(frame_number, pose, track, track_is_good, track_frame)
            for track_frame in self.track_frames
        )

    def distance(self, frame_number, pose, track, track_is_good, track_frame=-1):  # pylint: disable=unused-argument
        last_track_frame = track.frame_pose[-1][0]
        skipped_frames = frame_number - last_track_frame - 1
        assert skipped_frames >= 0
        if skipped_frames > 12:
            return 1000.0

        # correct track_frame with skipped_frames
        track_frame += skipped_frames
        if track_frame > -1:
            return 1000.0
        if len(track.frame_pose) < -track_frame:
            return 1000.0

        pose1 = pose.data[self.valid_keypoints]
        pose2 = track.frame_pose[track_frame][1].data[self.valid_keypoints]

        kps_distances = np.linalg.norm(pose2[:, :2] - pose1[:, :2], axis=1)
        kps_distances = np.clip(kps_distances, 0.0, self.invisible_penalty)
        kps_distances[pose1[:, 2] < 0.05] = self.invisible_penalty
        kps_distances[pose2[:, 2] < 0.05] = self.invisible_penalty
        kps_distance = np.mean(kps_distances)

        return kps_distance
