import numpy as np

import openpifpaf

from .posetrack2018 import KEYPOINTS, SKELETON, SIGMAS, UPRIGHT_POSE


def main():
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    openpifpaf.show.KeypointPainter.line_width = 6
    openpifpaf.show.KeypointPainter.monocolor_connections = False
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(UPRIGHT_POSE[:, 0]) - np.min(UPRIGHT_POSE[:, 0]))
        * (np.max(UPRIGHT_POSE[:, 1]) - np.min(UPRIGHT_POSE[:, 1]))
    )

    ann_posetrack = openpifpaf.Annotation(KEYPOINTS, SKELETON)
    ann_posetrack.set(UPRIGHT_POSE, np.array(SIGMAS) * scale, fixed_score='')
    with openpifpaf.show.Canvas.annotation(
            ann_posetrack, filename='docs/skeleton_posetrack.png') as ax:
        keypoint_painter.annotation(ax, ann_posetrack)

    upright_pose_2tracking = np.concatenate([
        UPRIGHT_POSE,
        0.9 * UPRIGHT_POSE + np.array([-1.5, 1.5, 0.0]),
    ])
    sigmas_2tracking = np.concatenate([np.array(SIGMAS) * scale, 0.8 * np.array(SIGMAS) * scale])
    tracking2_skeleton = np.concatenate([
        np.array(SKELETON) + 17,
        np.array([(j, j + 17) for j in range(1, 18)]),
        np.array(SKELETON),
    ])
    ann_tracking = openpifpaf.Annotation(KEYPOINTS + KEYPOINTS, tracking2_skeleton)
    ann_tracking.set(
        upright_pose_2tracking,
        sigmas_2tracking,
        fixed_score='',
    )
    with openpifpaf.show.Canvas.annotation(
            ann_tracking, filename='docs/skeleton_tracking.png') as ax:
        keypoint_painter.annotation(ax, ann_tracking)

    tracking_skeleton_forward = np.concatenate([
        np.array([(j, j + 17) for j in range(1, 18)]),
        np.array(SKELETON),
    ])
    ann_tracking_forward = openpifpaf.Annotation(KEYPOINTS + KEYPOINTS, tracking_skeleton_forward)
    ann_tracking_forward.set(
        upright_pose_2tracking,
        sigmas_2tracking,
        fixed_score='',
    )
    with openpifpaf.show.Canvas.annotation(
            ann_tracking_forward, filename='docs/skeleton_tracking_forward.png') as ax:
        keypoint_painter.annotation(ax, ann_tracking_forward)

    # COCO
    coco_keypoints = openpifpaf.plugins.coco.constants.COCO_KEYPOINTS
    coco_skeleton = openpifpaf.plugins.coco.constants.COCO_PERSON_SKELETON
    coco_skeleton_forward = np.concatenate([
        np.array([(j, j + 17) for j in range(1, 18)]),
        np.array(coco_skeleton),
    ])
    coco_upright_pose_2tracking = np.concatenate([
        openpifpaf.plugins.coco.constants.COCO_UPRIGHT_POSE,
        0.9 * openpifpaf.plugins.coco.constants.COCO_UPRIGHT_POSE + np.array([-1.5, 1.5, 0.0]),
    ])
    coco_sigmas_2tracking = np.concatenate([
        np.array(openpifpaf.plugins.coco.constants.COCO_PERSON_SIGMAS) * scale,
        0.8 * np.array(openpifpaf.plugins.coco.constants.COCO_PERSON_SIGMAS) * scale,
    ])
    ann_coco_tracking_forward = openpifpaf.Annotation(
        coco_keypoints + coco_keypoints, coco_skeleton_forward)
    ann_coco_tracking_forward.set(
        coco_upright_pose_2tracking,
        coco_sigmas_2tracking,
        fixed_score='',
    )
    with openpifpaf.show.Canvas.annotation(
            ann_coco_tracking_forward, filename='docs/coco_skeleton_forward.png') as ax:
        keypoint_painter.annotation(ax, ann_coco_tracking_forward)

    # overview
    with openpifpaf.show.Canvas.blank(
            'docs/skeleton_overview.png', figsize=(12, 6), ncols=4) as (ax1, ax2, ax3, ax4):
        ax1.set_axis_off()
        ax1.set_aspect('equal')
        keypoint_painter.annotation(ax1, ann_posetrack)

        ax2.set_axis_off()
        ax2.set_aspect('equal')
        keypoint_painter.annotation(ax2, ann_tracking)

        ax3.set_axis_off()
        ax3.set_aspect('equal')
        keypoint_painter.annotation(ax3, ann_tracking_forward)

        ax4.set_axis_off()
        ax4.set_aspect('equal')
        keypoint_painter.annotation(ax4, ann_coco_tracking_forward)


if __name__ == '__main__':
    main()
