import logging
import numpy as np

LOG = logging.getLogger(__name__)


class Crafted:
    """Compute hand crafted distance between a track and a new pose candidate.

    Ingredients:
    * compare to last pose in track and previous poses in case of temp corruption
    * require low distance for points that have high confidence in both poses (>=3 points)
    * "high confidence" is a dynamic measure dependent on the past track
    * penalize crappy tracks
    * penalize crappy poses
    """

    invisible_penalty = 110.0

    def __init__(self):
        self.valid_keypoints = None

    def __call__(self, frame_number, pose, track, track_is_good):
        return min((
            self.distance(frame_number, pose, track, track_is_good),
            self.distance(frame_number, pose, track, track_is_good, -4),
            self.distance(frame_number, pose, track, track_is_good, -8),
            self.distance(frame_number, pose, track, track_is_good, -12),
        ))

    # pylint: disable=too-many-return-statements,too-many-branches
    def distance(self, frame_number, pose, track, track_is_good, track_frame=-1):
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

        keypoint_scores = pose1[:, 2] * pose2[:, 2]
        kps_order = np.argsort(keypoint_scores)[::-1]
        if pose1[kps_order[2], 2] < 0.05 or pose2[kps_order[2], 2] < 0.05:
            return 1000.0
        pose1_center = np.mean(pose1[kps_order[:3], :2], axis=0)
        pose1_centered = np.copy(pose1)
        pose1_centered[:, :2] -= pose1_center
        pose2_center = np.mean(pose2[kps_order[:3], :2], axis=0)
        pose2_centered = np.copy(pose2)
        pose2_centered[:, :2] -= pose2_center
        center_distance = np.linalg.norm(pose2_center - pose1_center)

        kps_distances = np.linalg.norm(pose2_centered[:, :2] - pose1_centered[:, :2], axis=1)
        kps_distances = np.clip(kps_distances, 0.0, self.invisible_penalty)
        kps_distances[pose1[:, 2] < 0.05] = self.invisible_penalty
        kps_distances[pose2[:, 2] < 0.05] = self.invisible_penalty
        kps_distance_centered = np.mean(kps_distances)

        crappy_track_penalty = 0.0
        if len(track.frame_pose) < 4:
            crappy_track_penalty = 40.0
        elif len(track.frame_pose) < 8:
            crappy_track_penalty = 8.0
        if not track_is_good:
            crappy_track_penalty = max(crappy_track_penalty, 8.0)

        crappy_pose_penalty = 0.0
        if pose.score() < 0.2:
            crappy_pose_penalty = 40.0
        elif pose.score() < 0.5:
            crappy_pose_penalty = 8.0

        # skipping frames cost
        skipped_frame_cost = 40.0 if track_frame < -1 else 0.0

        return (
            center_distance / 10.0
            + kps_distance_centered
            + crappy_track_penalty
            + crappy_pose_penalty
            + skipped_frame_cost
        )
