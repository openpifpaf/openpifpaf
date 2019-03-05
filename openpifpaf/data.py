
COCO_PERSON_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]


KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]


COCO_KEYPOINTS = [
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
]


HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


DENSER_COCO_PERSON_SKELETON = [
    (1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5),
    (1, 6), (1, 7), (2, 6), (3, 7),
    (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
    (6, 12), (7, 13), (6, 13), (7, 12), (12, 13),
    (6, 8), (7, 9), (8, 10), (9, 11), (6, 10), (7, 11),
    (8, 9), (10, 11),
    (10, 12), (11, 13),
    (10, 14), (11, 15),
    (14, 12), (15, 13), (12, 15), (13, 14),
    (12, 16), (13, 17),
    (16, 14), (17, 15), (14, 17), (15, 16),
    (14, 15), (16, 17),
]


COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]


def draw_skeletons():
    import numpy as np
    from . import show
    coordinates = np.array([[
        [0.0, 9.3, 2.0],  # 'nose',            # 1
        [-0.5, 9.7, 2.0],  # 'left_eye',        # 2
        [0.5, 9.7, 2.0],  # 'right_eye',       # 3
        [-1.0, 9.5, 2.0],  # 'left_ear',        # 4
        [1.0, 9.5, 2.0],  # 'right_ear',       # 5
        [-2.0, 8.0, 2.0],  # 'left_shoulder',   # 6
        [2.0, 8.0, 2.0],  # 'right_shoulder',  # 7
        [-2.5, 6.0, 2.0],  # 'left_elbow',      # 8
        [2.5, 6.2, 2.0],  # 'right_elbow',     # 9
        [-2.5, 4.0, 2.0],  # 'left_wrist',      # 10
        [2.5, 4.2, 2.0],  # 'right_wrist',     # 11
        [-1.8, 4.0, 2.0],  # 'left_hip',        # 12
        [1.8, 4.0, 2.0],  # 'right_hip',       # 13
        [-2.0, 2.0, 2.0],  # 'left_knee',       # 14
        [2.0, 2.1, 2.0],  # 'right_knee',      # 15
        [-2.0, 0.0, 2.0],  # 'left_ankle',      # 16
        [2.0, 0.1, 2.0],  # 'right_ankle',     # 17
    ]])

    keypoint_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                            markersize=1, linewidth=6)

    with show.canvas('docs/skeleton_coco.png', figsize=(2, 5)) as ax:
        ax.set_axis_off()
        keypoint_painter.skeleton = COCO_PERSON_SKELETON
        keypoint_painter.keypoints(ax, coordinates)

    with show.canvas('docs/skeleton_kinematic_tree.png', figsize=(2, 5)) as ax:
        ax.set_axis_off()
        keypoint_painter.skeleton = KINEMATIC_TREE_SKELETON
        keypoint_painter.keypoints(ax, coordinates)

    with show.canvas('docs/skeleton_dense.png', figsize=(2, 5)) as ax:
        ax.set_axis_off()
        keypoint_painter.skeleton = DENSER_COCO_PERSON_SKELETON
        keypoint_painter.keypoints(ax, coordinates)


if __name__ == '__main__':
    draw_skeletons()
