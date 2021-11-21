import numpy as np


COCO_PERSON_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]


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


COCO_UPRIGHT_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
    [1.4, 2.1, 2.0],  # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
])


COCO_DAVINCI_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-3.3, 9.0, 2.0],  # 'left_elbow',      # 8
    [3.3, 9.2, 2.0],  # 'right_elbow',     # 9
    [-4.5, 10.5, 2.0],  # 'left_wrist',      # 10
    [4.5, 10.7, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-2.0, 2.0, 2.0],  # 'left_knee',       # 14
    [2.0, 2.1, 2.0],  # 'right_knee',      # 15
    [-2.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [2.4, 0.1, 2.0],  # 'right_ankle',     # 17
])


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


DENSER_COCO_PERSON_CONNECTIONS = [
    c
    for c in DENSER_COCO_PERSON_SKELETON
    if c not in COCO_PERSON_SKELETON
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


COCO_PERSON_SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(COCO_KEYPOINTS) - 3)


COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]


def draw_skeletons(pose):
    import openpifpaf  # pylint: disable=import-outside-toplevel
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    ann = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS,
                                skeleton=COCO_PERSON_SKELETON,
                                score_weights=COCO_PERSON_SCORE_WEIGHTS)
    ann.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, filename='docs/skeleton_coco.png') as ax:
        keypoint_painter.annotation(ax, ann)

    ann_kin = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS,
                                    skeleton=KINEMATIC_TREE_SKELETON,
                                    score_weights=COCO_PERSON_SCORE_WEIGHTS)
    ann_kin.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann_kin, filename='docs/skeleton_kinematic_tree.png') as ax:
        keypoint_painter.annotation(ax, ann_kin)

    ann_dense = openpifpaf.Annotation(keypoints=COCO_KEYPOINTS,
                                      skeleton=DENSER_COCO_PERSON_SKELETON,
                                      score_weights=COCO_PERSON_SCORE_WEIGHTS)
    ann_dense.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, ann_bg=ann_dense, filename='docs/skeleton_dense.png') as ax:
        keypoint_painter.annotation(ax, ann_dense)


def print_associations():
    for j1, j2 in COCO_PERSON_SKELETON:
        print(COCO_KEYPOINTS[j1 - 1], '-', COCO_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(COCO_UPRIGHT_POSE)
